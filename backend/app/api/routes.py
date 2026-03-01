"""API routes for QA Evaluation Studio."""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import defaultdict
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.schemas import (
    IngestRequest, IngestResponse, EvaluateRunRequest, EvaluateRunResponse,
    ABExperimentRequest, ABExperimentResponse, EvalItemCreate, EvalItemInDB,
    EvalItemListResponse, ScenarioItemsResponse,
    EvaluationInDB, HumanReviewCreate, HumanReviewInDB,
    HumanQueueItem, ReportSummaryResponse, ExperimentInDB, DatasetSplit,
    DocumentMeta,
    # Phase 2: Pattern Analysis
    PatternAnalysisRequest, PatternAnalysisResult, FailurePattern,
    # Phase 4: Prompt Suggestions
    SuggestionGenerateRequest, PromptSuggestion,
    # Phase 5: Multi-Comparison
    MultiComparisonRequest, MultiComparisonSummary,
    # Phase 6: Approval Workflow
    PromptProposalCreate, PromptProposalInDB, ProposalStatus,
)
from ..models.database import init_db
from ..db.repository import (
    EvalItemRepository, EvaluationRepository, JudgeOutputRepository,
    HumanQueueRepository, HumanReviewRepository, ExperimentRepository,
    DocumentRepository, FailurePatternRepository, ProposalRepository,
)
from ..providers import get_provider
from ..rag.indexer import RAGIndexer
from ..rag.retriever import RAGRetriever
from ..services.instrumentation import LangfuseInstrumentation
from ..services.pipeline import EvaluationPipeline
from ..services.experiment import ExperimentService
from ..services.pattern_analyzer import PatternAnalyzer
from ..services.prompt_suggester import PromptSuggester
from ..services.multi_compare import MultiCompareService
from ..services.approval_workflow import ApprovalWorkflow

router = APIRouter()
settings = get_settings()


def _validate_model_version(model_version: str) -> None:
    """Reject 'mock' model_version when using a real LLM provider."""
    if settings.LLM_PROVIDER != "mock" and model_version == "mock":
        raise HTTPException(
            status_code=422,
            detail=f"model_version='mock' is invalid when LLM_PROVIDER='{settings.LLM_PROVIDER}'. "
                   f"Use a real model id (e.g. 'gpt-4o-mini').",
        )

# Initialize database
engine, SessionLocal = init_db(settings.DATABASE_URL)


def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_instrumentation(db: Session = Depends(get_db)) -> LangfuseInstrumentation:
    """Get Langfuse instrumentation."""
    return LangfuseInstrumentation(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_BASE_URL,
        db_session=db,
    )


_rag_retriever: Optional[RAGRetriever] = None


def get_rag_retriever() -> RAGRetriever:
    """Get cached RAG retriever (built once at startup or on first request)."""
    global _rag_retriever
    if _rag_retriever is None:
        use_mock = settings.LLM_PROVIDER == "mock" or not settings.OPENAI_API_KEY
        indexer = RAGIndexer(
            docs_path=settings.DOCS_PATH,
            vector_store_path=settings.VECTOR_STORE_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            use_mock=use_mock,
        )
        indexer.load_index() or indexer.build_index()
        _rag_retriever = RAGRetriever(indexer)
    return _rag_retriever


def _resolve_ingest_path(raw_path: str) -> Path:
    """Resolve and validate ingest path against approved project data roots."""
    project_root = Path(__file__).resolve().parents[3]
    allowed_roots = [
        (project_root / "backend" / "data").resolve(),
        (project_root / "sample_data").resolve(),
    ]

    candidate = Path(raw_path).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()

    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail=(
                "File path is not allowed. Use files under "
                "'backend/data' or 'sample_data'."
            ),
        )

    return candidate


# ==================== INGEST ====================

async def _parse_and_ingest(
    data_lines: List[str],
    split: DatasetSplit,
    db: Session
) -> IngestResponse:
    """Parse JSONL lines and ingest eval items."""
    item_repo = EvalItemRepository(db)
    errors = []
    items_to_create = []

    for i, line in enumerate(data_lines):
        try:
            data = json.loads(line)

            # Supports standard Q/A, UltraFeedback, and legacy ticket formats
            system_prompt = data.get("system_prompt", data.get("system", ""))
            question = data.get("question", data.get("instruction", data.get("query", "")))
            response = data.get("response", data.get("output", data.get("answer", "")))

            # Legacy ticket format: conversation[0].content → question,
            # candidate_response → response
            if not question and "conversation" in data:
                conv = data["conversation"]
                if isinstance(conv, list) and conv:
                    question = conv[0].get("content", "") if isinstance(conv[0], dict) else str(conv[0])
            if not response and "candidate_response" in data:
                response = data["candidate_response"]

            if not question or not response:
                errors.append(f"Line {i+1}: Missing question or response field")
                continue

            item = EvalItemCreate(
                external_id=data.get("id", data.get("external_id")),
                system_prompt=system_prompt or None,
                question=question,
                response=response,
                metadata=data.get("metadata"),
                scenario_id=data.get("scenario_id"),
                candidate_source=data.get("candidate_source"),
                split=split,
            )
            items_to_create.append(item)

        except json.JSONDecodeError as e:
            errors.append(f"Line {i+1}: JSON parse error - {e}")
        except Exception as e:
            errors.append(f"Line {i+1}: {str(e)}")

    # Batch insert all items in a single transaction
    ingested = 0
    if items_to_create:
        ingested = item_repo.create_batch(items_to_create)

    return IngestResponse(
        ingested_count=ingested,
        split=split,
        errors=errors[:10],
    )


@router.post("/ingest/batch", response_model=IngestResponse)
async def ingest_batch(
    request: IngestRequest,
    db: Session = Depends(get_db)
):
    """Ingest batch of eval items from server file path."""
    file_path = _resolve_ingest_path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    with open(file_path) as f:
        if str(file_path).endswith(".jsonl"):
            data_lines = f.read().strip().split("\n")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .jsonl")

    return await _parse_and_ingest(data_lines, request.split, db)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    split: DatasetSplit = Form(DatasetSplit.DEV),
    db: Session = Depends(get_db)
):
    """Ingest batch of eval items from file upload."""
    content = await file.read()
    content_str = content.decode("utf-8")

    if file.filename.endswith(".jsonl"):
        data_lines = content_str.strip().split("\n")
    elif file.filename.endswith(".csv"):
        import csv
        import io
        data_lines = []
        reader = csv.DictReader(io.StringIO(content_str))
        for row in reader:
            data_lines.append(json.dumps(row))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use .jsonl or .csv")

    return await _parse_and_ingest(data_lines, split, db)


# ==================== EVALUATE ====================

@router.post("/evaluate/run", response_model=EvaluateRunResponse)
async def evaluate_run(
    request: EvaluateRunRequest,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation)
):
    """Run evaluation on items."""
    _validate_model_version(request.model_version)

    # Cost safety: require explicit limit or item_ids when using a real provider
    if settings.LLM_PROVIDER != "mock" and not request.item_ids and not request.limit:
        raise HTTPException(
            status_code=422,
            detail="'limit' or 'item_ids' is required when LLM_PROVIDER is not 'mock' to prevent runaway API costs.",
        )

    item_repo = EvalItemRepository(db)
    retriever = get_rag_retriever()

    # Get provider
    provider = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=request.model_version or settings.LLM_MODEL,
        instrumentation=instrumentation,
    )

    # Create pipeline
    pipeline = EvaluationPipeline(
        provider=provider,
        retriever=retriever,
        instrumentation=instrumentation,
        db_session=db,
    )

    # Get items — respect explicit limit, otherwise process all items in the split
    eval_repo = EvaluationRepository(db)
    if request.item_ids:
        items = [i for iid in request.item_ids if (i := item_repo.get(iid))]
    else:
        # First get total count, then fetch up to limit (or all)
        _, total = item_repo.get_all(split=request.dataset_split, page=1, page_size=1)
        fetch_size = min(request.limit, total) if request.limit else total
        items, _ = item_repo.get_all(split=request.dataset_split, page=1, page_size=max(fetch_size, 1))

    # Skip items that already have an evaluation for this version triple
    already_evaluated = eval_repo.get_evaluated_item_ids(
        prompt_version=request.prompt_version,
        model_version=request.model_version,
        docs_version=request.docs_version,
    )
    items = [item for item in items if item.id not in already_evaluated]

    # Process items
    results = []
    for item in items:
        result = await pipeline.process_item(
            item,
            prompt_version=request.prompt_version,
            model_version=request.model_version,
            docs_version=request.docs_version,
            sampling_config=request.sampling_config,
        )
        results.append(result)

    # Aggregate results
    gate_fail_count = sum(1 for r in results if r.get("gate_failed"))
    human_queue_count = sum(1 for r in results if r.get("human_queued"))

    tag_counts = defaultdict(int)
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)

    for r in results:
        for tag in r.get("tags", []):
            tag_counts[tag] += 1
        for score_type, score in r.get("scores", {}).items():
            score_sums[score_type] += score
            score_counts[score_type] += 1

    avg_scores = {k: score_sums[k] / score_counts[k] for k in score_sums if score_counts[k] > 0}

    # Flush Langfuse traces
    instrumentation.flush()

    return EvaluateRunResponse(
        processed_count=len(results),
        gate_fail_count=gate_fail_count,
        human_queue_count=human_queue_count,
        top_tags=dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
        avg_scores=avg_scores,
    )


# ==================== EXPERIMENT ====================

@router.post("/experiment/ab", response_model=ABExperimentResponse)
async def run_ab_experiment(
    request: ABExperimentRequest,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation)
):
    """Run A/B experiment."""
    _validate_model_version(request.config_a.model_version)
    _validate_model_version(request.config_b.model_version)

    # Cost safety: require explicit limit or item_ids when using a real provider
    if settings.LLM_PROVIDER != "mock" and not request.item_ids and not request.limit:
        raise HTTPException(
            status_code=422,
            detail="'limit' or 'item_ids' is required when LLM_PROVIDER is not 'mock' to prevent runaway API costs.",
        )

    retriever = get_rag_retriever()

    # Get providers (can be same or different)
    provider_a = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=request.config_a.model_version,
        instrumentation=instrumentation,
    )
    provider_b = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=request.config_b.model_version,
        instrumentation=instrumentation,
    )

    # Create pipelines
    pipeline_a = EvaluationPipeline(
        provider=provider_a,
        retriever=retriever,
        instrumentation=instrumentation,
        db_session=db,
    )
    pipeline_b = EvaluationPipeline(
        provider=provider_b,
        retriever=retriever,
        instrumentation=instrumentation,
        db_session=db,
    )

    # Run experiment
    exp_service = ExperimentService(
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        instrumentation=instrumentation,
        db_session=db,
    )

    name = request.name or f"AB_{request.config_a.prompt_version}_vs_{request.config_b.prompt_version}"

    experiment = await exp_service.run_experiment(
        name=name,
        dataset_split=request.dataset_split,
        docs_version=request.docs_version,
        config_a=request.config_a,
        config_b=request.config_b,
        sampling_config=request.sampling_config,
        item_ids=request.item_ids,
        limit=request.limit,
    )

    return ABExperimentResponse(
        experiment_id=experiment.id,
        summary=experiment.summary,
    )


# ==================== ITEMS ====================

@router.get("/items", response_model=EvalItemListResponse)
async def list_items(
    split: Optional[DatasetSplit] = None,
    scenario_id: Optional[str] = None,
    candidate_source: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List eval items with pagination and optional filters."""
    repo = EvalItemRepository(db)
    items, total = repo.get_all(
        split=split,
        scenario_id=scenario_id,
        candidate_source=candidate_source,
        page=page,
        page_size=page_size,
    )

    return EvalItemListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/items/scenario/{scenario_id}", response_model=ScenarioItemsResponse)
async def get_items_by_scenario(
    scenario_id: str,
    db: Session = Depends(get_db)
):
    """Get all candidate items for a scenario."""
    repo = EvalItemRepository(db)
    items = repo.get_by_scenario(scenario_id)
    return ScenarioItemsResponse(scenario_id=scenario_id, items=items, count=len(items))


@router.get("/items/{item_id}")
async def get_item(
    item_id: str,
    db: Session = Depends(get_db)
):
    """Get eval item by ID with evaluations."""
    item_repo = EvalItemRepository(db)
    eval_repo = EvaluationRepository(db)
    judge_repo = JudgeOutputRepository(db)

    item = item_repo.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    evaluations = eval_repo.get_by_item(item_id)

    # Enrich evaluations with judge outputs
    enriched_evals = []
    for eval in evaluations:
        judge = judge_repo.get_by_evaluation(eval.id)
        enriched_evals.append({
            **eval.model_dump(),
            "judge_output": judge.model_dump() if judge else None,
        })

    return {
        "item": item.model_dump(),
        "evaluations": enriched_evals,
    }


# ==================== EVALUATIONS ====================

@router.get("/evaluations")
async def list_evaluations(
    item_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    model_version: Optional[str] = None,
    docs_version: str = Query("v1", description="Docs version (defaults to 'v1')"),
    db: Session = Depends(get_db)
):
    """List evaluations with optional filters."""
    repo = EvaluationRepository(db)

    if item_id:
        evaluations = repo.get_by_item(item_id)
    elif prompt_version and model_version:
        evaluations = repo.get_by_version(prompt_version, model_version, docs_version)
    else:
        raise HTTPException(status_code=400, detail="Provide item_id or version filters")

    return {"evaluations": [e.model_dump() for e in evaluations]}


# ==================== HUMAN QUEUE ====================

@router.get("/human/queue", response_model=List[HumanQueueItem])
async def get_human_queue(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get pending items in human review queue."""
    repo = HumanQueueRepository(db)
    return repo.get_pending(limit=limit, offset=offset)


@router.post("/human/review", response_model=HumanReviewInDB)
async def submit_human_review(
    review: HumanReviewCreate,
    db: Session = Depends(get_db)
):
    """Submit human review (gold label)."""
    review_repo = HumanReviewRepository(db)
    queue_repo = HumanQueueRepository(db)

    try:
        # Keep queue and review updates in a single transaction.
        marked = queue_repo.mark_reviewed(review.queue_item_id, commit=False)
        if not marked:
            db.rollback()
            raise HTTPException(status_code=404, detail="Queue item not found")

        created_review = review_repo.create(review, commit=False)
        db.commit()
        return created_review
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit human review: {e}")


# ==================== REPORTS ====================

@router.get("/reports/summary", response_model=ReportSummaryResponse)
async def get_report_summary(
    dataset_split: DatasetSplit,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get summary report for evaluations."""
    item_repo = EvalItemRepository(db)
    queue_repo = HumanQueueRepository(db)

    stats = item_repo.get_summary_stats(dataset_split)
    total_evals = stats["total_evaluations"]
    pending_count = queue_repo.count_pending(split=dataset_split)

    return ReportSummaryResponse(
        dataset_split=dataset_split,
        date_range=f"{start_date} to {end_date}" if start_date else None,
        total_evaluations=total_evals,
        gate_fail_rate=stats["gate_fail_count"] / total_evals if total_evals > 0 else 0,
        avg_scores=stats.get("avg_scores", {}),
        tag_distribution=dict(sorted(stats.get("tag_counts", {}).items(), key=lambda x: -x[1])),
        human_queue_stats={"pending": pending_count},
    )


# ==================== EXPERIMENTS ====================

@router.get("/experiments")
async def list_experiments(db: Session = Depends(get_db)):
    """List all experiments."""
    repo = ExperimentRepository(db)
    experiments = repo.get_all()
    return {"experiments": [e.model_dump() for e in experiments]}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Get experiment by ID."""
    repo = ExperimentRepository(db)
    experiment = repo.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.model_dump()


# ==================== DOCUMENTS ====================

@router.get("/documents")
async def list_documents(
    version: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List documents in the RAG index."""
    repo = DocumentRepository(db)
    if version:
        docs = repo.get_by_version(version)
    else:
        docs = repo.get_all()
    return {"documents": [d.model_dump() for d in docs]}


@router.post("/documents/reindex")
async def reindex_documents(db: Session = Depends(get_db)):
    """Rebuild the RAG index and sync documents to DB."""
    use_mock = settings.LLM_PROVIDER == "mock" or not settings.OPENAI_API_KEY
    indexer = RAGIndexer(
        docs_path=settings.DOCS_PATH,
        vector_store_path=settings.VECTOR_STORE_PATH,
        embedding_model=settings.EMBEDDING_MODEL,
        use_mock=use_mock,
    )
    success = indexer.build_index()

    # Persist indexed documents to DB so /documents listing stays in sync
    if success:
        doc_repo = DocumentRepository(db)
        for doc in indexer.documents:
            existing = doc_repo.get(doc["doc_id"])
            if not existing:
                doc_repo.create(
                    DocumentMeta(
                        doc_id=doc["doc_id"],
                        title=doc["title"],
                        source_url=doc.get("source_url"),
                        version=doc.get("version", "v1"),
                        tags=doc.get("tags", []),
                        category=doc["category"],
                    ),
                    content=doc["content"],
                )

    # Update cached retriever
    global _rag_retriever
    _rag_retriever = RAGRetriever(indexer)

    return {"success": success, "document_count": len(indexer.documents)}


# ==================== HEALTH ====================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "langfuse_enabled": bool(settings.LANGFUSE_PUBLIC_KEY),
        "llm_provider": settings.LLM_PROVIDER,
    }


# ==================== PROMPTS (Phase 1) ====================

@router.get("/prompts")
async def list_prompts(
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """List all prompts from Langfuse registry."""
    prompts = instrumentation.list_prompts()
    return {"prompts": prompts}


@router.post("/prompts")
async def create_prompt(
    name: str = Body(...),
    prompt: str = Body(...),
    labels: Optional[List[str]] = Body(None),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Register a prompt in Langfuse."""
    result = instrumentation.create_prompt(name=name, prompt=prompt, labels=labels)
    if result is None:
        raise HTTPException(status_code=503, detail="Langfuse not configured or unavailable")
    return result


@router.post("/prompts/{name}/label")
async def update_prompt_label(
    name: str,
    version: int = Body(...),
    new_label: str = Body(...),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Update a prompt version label (e.g., promote to 'production')."""
    success = instrumentation.update_prompt_label(name=name, version=version, new_label=new_label)
    if not success:
        raise HTTPException(status_code=503, detail="Langfuse not configured or update failed")
    return {"success": True, "name": name, "version": version, "label": new_label}


# ==================== PATTERN ANALYSIS (Phase 2) ====================

@router.post("/analysis/patterns", response_model=PatternAnalysisResult)
async def run_pattern_analysis(
    request: PatternAnalysisRequest,
    db: Session = Depends(get_db),
):
    """Run failure pattern analysis across stored evaluations."""
    analyzer = PatternAnalyzer(db_session=db)
    return await analyzer.analyze(
        prompt_version=request.prompt_version,
        model_version=request.model_version,
        min_frequency=request.min_frequency,
        top_k=request.top_k,
    )


@router.get("/analysis/patterns/latest", response_model=List[FailurePattern])
async def get_latest_patterns(
    top_k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get patterns from the most recent analysis run."""
    repo = FailurePatternRepository(db)
    return repo.get_latest(top_k=top_k)


# ==================== SUGGESTIONS (Phase 4) ====================

@router.post("/suggestions/generate", response_model=List[PromptSuggestion])
async def generate_suggestions(
    request: SuggestionGenerateRequest,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Generate prompt improvement suggestions from failure patterns."""
    provider = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        instrumentation=instrumentation,
    )
    suggester = PromptSuggester(
        provider=provider,
        db_session=db,
        instrumentation=instrumentation,
    )
    return await suggester.generate_suggestions(request)


@router.get("/suggestions/latest", response_model=List[PromptSuggestion])
async def get_latest_suggestions(
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Get the most recently generated prompt suggestions."""
    provider = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        instrumentation=instrumentation,
    )
    suggester = PromptSuggester(
        provider=provider,
        db_session=db,
        instrumentation=instrumentation,
    )
    return suggester.get_latest_suggestions(top_k=top_k)


# ==================== MULTI-COMPARISON (Phase 5) ====================

@router.post("/experiment/multi", response_model=MultiComparisonSummary)
async def run_multi_comparison(
    request: MultiComparisonRequest,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Run N-way config comparison across a dataset split."""
    for cfg in request.configs:
        _validate_model_version(cfg.model_version)

    retriever = get_rag_retriever()

    def provider_factory(prompt_version: str, model_version: str):
        return get_provider(
            settings.LLM_PROVIDER,
            api_key=settings.OPENAI_API_KEY,
            model=model_version,
            instrumentation=instrumentation,
        )

    service = MultiCompareService(db_session=db, instrumentation=instrumentation)
    return await service.run_comparison(
        request=request,
        provider_factory=provider_factory,
        retriever=retriever,
    )


@router.get("/experiment/multi/{experiment_id}/results")
async def get_multi_comparison_results(
    experiment_id: str,
    db: Session = Depends(get_db),
):
    """Get per-item results for a multi-comparison experiment."""
    from ..db.repository import MultiComparisonRepository
    repo = MultiComparisonRepository(db)
    results = repo.get_by_experiment(experiment_id)
    return {"experiment_id": experiment_id, "results": results}


# ==================== APPROVAL WORKFLOW (Phase 6) ====================

@router.get("/proposals", response_model=List[PromptProposalInDB])
async def list_proposals(
    status: Optional[ProposalStatus] = None,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List prompt proposals, optionally filtered by status."""
    workflow = ApprovalWorkflow(db_session=db)
    return workflow.list_proposals(status=status, limit=limit)


@router.post("/proposals", response_model=PromptProposalInDB)
async def create_proposal(
    data: PromptProposalCreate,
    db: Session = Depends(get_db),
):
    """Create a new prompt proposal (starts in PENDING state)."""
    workflow = ApprovalWorkflow(db_session=db)
    return workflow.create_proposal(data)


@router.get("/proposals/{proposal_id}", response_model=PromptProposalInDB)
async def get_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
):
    """Get a proposal by ID."""
    workflow = ApprovalWorkflow(db_session=db)
    proposal = workflow.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


@router.post("/proposals/{proposal_id}/test", response_model=PromptProposalInDB)
async def start_proposal_test(
    proposal_id: str,
    experiment_id: str = Body(..., embed=True),
    db: Session = Depends(get_db),
):
    """Start A/B test for a proposal (PENDING → TESTING)."""
    workflow = ApprovalWorkflow(db_session=db)
    try:
        return workflow.start_test(proposal_id, experiment_id=experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/proposals/{proposal_id}/approve", response_model=PromptProposalInDB)
async def approve_proposal(
    proposal_id: str,
    improvement_metrics: Optional[Dict[str, Any]] = Body(None),
    db: Session = Depends(get_db),
):
    """Approve a proposal (TESTING → APPROVED)."""
    workflow = ApprovalWorkflow(db_session=db)
    try:
        return workflow.approve(proposal_id, improvement_metrics=improvement_metrics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/proposals/{proposal_id}/reject", response_model=PromptProposalInDB)
async def reject_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
):
    """Reject a proposal (any state → REJECTED)."""
    workflow = ApprovalWorkflow(db_session=db)
    try:
        return workflow.reject(proposal_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/proposals/{proposal_id}/deploy", response_model=PromptProposalInDB)
async def deploy_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Deploy an approved proposal to Langfuse production label (APPROVED → DEPLOYED)."""
    workflow = ApprovalWorkflow(db_session=db, instrumentation=instrumentation)
    try:
        return workflow.deploy(proposal_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

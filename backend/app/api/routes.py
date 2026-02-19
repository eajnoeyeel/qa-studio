"""API routes for CS QA Studio."""
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
    ABExperimentRequest, ABExperimentResponse, TicketCreate, TicketInDB,
    TicketListResponse, EvaluationInDB, HumanReviewCreate, HumanReviewInDB,
    HumanQueueItem, ReportSummaryResponse, ExperimentInDB, Message, DatasetSplit,
    DocumentMeta
)
from ..models.database import init_db
from ..db.repository import (
    TicketRepository, EvaluationRepository, JudgeOutputRepository,
    HumanQueueRepository, HumanReviewRepository, ExperimentRepository,
    DocumentRepository
)
from ..providers import get_provider
from ..rag.indexer import RAGIndexer
from ..rag.retriever import RAGRetriever
from ..services.instrumentation import LangfuseInstrumentation
from ..services.pipeline import EvaluationPipeline
from ..services.experiment import ExperimentService

router = APIRouter()
settings = get_settings()

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
        host=settings.LANGFUSE_HOST,
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


# ==================== INGEST ====================

async def _parse_and_ingest(
    data_lines: List[str],
    split: DatasetSplit,
    db: Session
) -> IngestResponse:
    """Parse JSONL lines and ingest tickets."""
    ticket_repo = TicketRepository(db)
    errors = []
    tickets_to_create = []

    for i, line in enumerate(data_lines):
        try:
            data = json.loads(line)

            # Convert to ticket format
            messages = []
            if "conversation" in data:
                for msg in data["conversation"]:
                    messages.append(Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                    ))
            elif "messages" in data:
                for msg in data["messages"]:
                    messages.append(Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                    ))
            elif "query" in data and "response" in data:
                messages.append(Message(role="user", content=data["query"]))
                messages.append(Message(role="assistant", content=data["response"]))

            candidate_response = data.get("candidate_response", data.get("response", ""))

            if not messages:
                errors.append(f"Line {i+1}: No conversation found")
                continue

            ticket = TicketCreate(
                external_id=data.get("id", data.get("external_id")),
                conversation=messages,
                candidate_response=candidate_response,
                metadata=data.get("metadata"),
                split=split,
            )
            tickets_to_create.append(ticket)

        except json.JSONDecodeError as e:
            errors.append(f"Line {i+1}: JSON parse error - {e}")
        except Exception as e:
            errors.append(f"Line {i+1}: {str(e)}")

    # Batch insert all tickets in a single transaction
    ingested = 0
    if tickets_to_create:
        ingested = ticket_repo.create_batch(tickets_to_create)

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
    """Ingest batch of tickets from server file path."""
    file_path = Path(request.file_path)
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
    """Ingest batch of tickets from file upload."""
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
    """Run evaluation on tickets."""
    ticket_repo = TicketRepository(db)
    retriever = get_rag_retriever()

    # Get provider
    provider = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
    )

    # Create pipeline
    pipeline = EvaluationPipeline(
        provider=provider,
        retriever=retriever,
        instrumentation=instrumentation,
        db_session=db,
    )

    # Get tickets
    if request.ticket_ids:
        tickets = [t for tid in request.ticket_ids if (t := ticket_repo.get(tid))]
    else:
        tickets, _ = ticket_repo.get_all(split=request.dataset_split, page=1, page_size=1000)

    # Process tickets
    results = []
    for ticket in tickets:
        result = await pipeline.process_ticket(
            ticket,
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
    retriever = get_rag_retriever()

    # Get providers (can be same or different)
    provider_a = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=request.config_a.model_version,
    )
    provider_b = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=request.config_b.model_version,
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
    )

    return ABExperimentResponse(
        experiment_id=experiment.id,
        summary=experiment.summary,
    )


# ==================== TICKETS ====================

@router.get("/tickets", response_model=TicketListResponse)
async def list_tickets(
    split: Optional[DatasetSplit] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List tickets with pagination."""
    repo = TicketRepository(db)
    tickets, total = repo.get_all(split=split, page=page, page_size=page_size)

    return TicketListResponse(
        tickets=tickets,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/tickets/{ticket_id}")
async def get_ticket(
    ticket_id: str,
    db: Session = Depends(get_db)
):
    """Get ticket by ID with evaluations."""
    ticket_repo = TicketRepository(db)
    eval_repo = EvaluationRepository(db)
    judge_repo = JudgeOutputRepository(db)

    ticket = ticket_repo.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    evaluations = eval_repo.get_by_ticket(ticket_id)

    # Enrich evaluations with judge outputs
    enriched_evals = []
    for eval in evaluations:
        judge = judge_repo.get_by_evaluation(eval.id)
        enriched_evals.append({
            **eval.model_dump(),
            "judge_output": judge.model_dump() if judge else None,
        })

    return {
        "ticket": ticket.model_dump(),
        "evaluations": enriched_evals,
    }


# ==================== EVALUATIONS ====================

@router.get("/evaluations")
async def list_evaluations(
    ticket_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    model_version: Optional[str] = None,
    docs_version: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List evaluations with optional filters."""
    repo = EvaluationRepository(db)

    if ticket_id:
        evaluations = repo.get_by_ticket(ticket_id)
    elif prompt_version and model_version:
        evaluations = repo.get_by_version(prompt_version, model_version, docs_version or "v1")
    else:
        raise HTTPException(status_code=400, detail="Provide ticket_id or version filters")

    return {"evaluations": [e.model_dump() for e in evaluations]}


# ==================== HUMAN QUEUE ====================

@router.get("/human/queue", response_model=List[HumanQueueItem])
async def get_human_queue(
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get pending items in human review queue."""
    repo = HumanQueueRepository(db)
    return repo.get_pending(limit=limit)


@router.post("/human/review", response_model=HumanReviewInDB)
async def submit_human_review(
    review: HumanReviewCreate,
    db: Session = Depends(get_db)
):
    """Submit human review (gold label)."""
    review_repo = HumanReviewRepository(db)
    queue_repo = HumanQueueRepository(db)

    # Mark queue item as reviewed
    queue_repo.mark_reviewed(review.queue_item_id)

    # Create review
    return review_repo.create(review)


# ==================== REPORTS ====================

@router.get("/reports/summary", response_model=ReportSummaryResponse)
async def get_report_summary(
    dataset_split: DatasetSplit,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get summary report for evaluations."""
    eval_repo = EvaluationRepository(db)
    judge_repo = JudgeOutputRepository(db)
    queue_repo = HumanQueueRepository(db)
    ticket_repo = TicketRepository(db)

    # Get tickets in split
    tickets, total = ticket_repo.get_all(split=dataset_split, page=1, page_size=10000)

    # Collect stats
    total_evals = 0
    gate_fails = 0
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for ticket in tickets:
        evals = eval_repo.get_by_ticket(ticket.id)
        for eval in evals:
            total_evals += 1
            judge = judge_repo.get_by_evaluation(eval.id)
            if judge:
                if not all(getattr(g, "passed", True) if not isinstance(g, dict) else g.get("passed", True) for g in judge.gates):
                    gate_fails += 1
                for score in judge.scores:
                    if isinstance(score, dict):
                        st, sv = score.get("score_type", ""), score.get("score", 0)
                    else:
                        st, sv = score.score_type, score.score
                    score_sums[st] += sv
                    score_counts[st] += 1
                for tag in judge.failure_tags:
                    tag_counts[tag] += 1

    # Calculate averages
    avg_scores = {k: score_sums[k] / score_counts[k] for k in score_sums if score_counts[k] > 0}

    # Queue stats
    pending_count = queue_repo.count_pending()

    return ReportSummaryResponse(
        dataset_split=dataset_split,
        date_range=f"{start_date} to {end_date}" if start_date else None,
        total_evaluations=total_evals,
        gate_fail_rate=gate_fails / total_evals if total_evals > 0 else 0,
        avg_scores=avg_scores,
        tag_distribution=dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
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

"""Evaluation endpoints."""
import asyncio
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...db.repository import EvalItemRepository, EvaluationRepository
from ...models.schemas import EvaluateRunRequest, EvaluateRunResponse
from ...providers import get_provider
from ...services.pipeline import EvaluationPipeline
from ...services.instrumentation import LangfuseInstrumentation
from ..deps import (
    SessionLocal,
    get_db,
    get_instrumentation,
    get_rag_retriever,
    settings,
    validate_model_version,
)

router = APIRouter()


@router.post("/evaluate/run", response_model=EvaluateRunResponse)
async def evaluate_run(
    request: EvaluateRunRequest,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Run evaluation on items."""
    validate_model_version(request.model_version)

    # Cost safety: require explicit limit or item_ids when using a real provider
    if settings.LLM_PROVIDER != "mock" and not request.item_ids and not request.limit:
        raise HTTPException(
            status_code=422,
            detail="'limit' or 'item_ids' is required when LLM_PROVIDER is not 'mock' to prevent runaway API costs.",
        )

    item_repo = EvalItemRepository(db)
    retriever = get_rag_retriever()

    provider = get_provider(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model=request.model_version or settings.LLM_MODEL,
        instrumentation=instrumentation,
    )

    pipeline = EvaluationPipeline(
        provider=provider,
        retriever=retriever,
        instrumentation=instrumentation,
        db_session=db,
        session_factory=SessionLocal,
    )

    eval_repo = EvaluationRepository(db)
    if request.item_ids:
        items = [i for iid in request.item_ids if (i := item_repo.get(iid))]
    else:
        _, total = item_repo.get_all(split=request.dataset_split, page=1, page_size=1)
        fetch_size = min(request.limit, total) if request.limit else total
        items, _ = item_repo.get_all(split=request.dataset_split, page=1, page_size=max(fetch_size, 1))

    already_evaluated = eval_repo.get_evaluated_item_ids(
        prompt_version=request.prompt_version,
        model_version=request.model_version,
        docs_version=request.docs_version,
    )
    items = [item for item in items if item.id not in already_evaluated]

    semaphore = asyncio.Semaphore(5)

    async def process_one(item):
        async with semaphore:
            return await pipeline.process_item(
                item,
                prompt_version=request.prompt_version,
                model_version=request.model_version,
                docs_version=request.docs_version,
                sampling_config=request.sampling_config,
            )

    results = await asyncio.gather(*[process_one(item) for item in items])

    error_count = sum(1 for r in results if r.get("error"))
    successful_results = [r for r in results if not r.get("error")]
    gate_fail_count = sum(1 for r in successful_results if r.get("gate_failed"))
    human_queue_count = sum(1 for r in successful_results if r.get("human_queued"))

    tag_counts = defaultdict(int)
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)

    for r in successful_results:
        for tag in r.get("tags", []):
            tag_counts[tag] += 1
        for score_type, score in r.get("scores", {}).items():
            score_sums[score_type] += score
            score_counts[score_type] += 1

    avg_scores = {k: score_sums[k] / score_counts[k] for k in score_sums if score_counts[k] > 0}

    instrumentation.flush()

    return EvaluateRunResponse(
        processed_count=len(successful_results),
        error_count=error_count,
        gate_fail_count=gate_fail_count,
        human_queue_count=human_queue_count,
        top_tags=dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
        avg_scores=avg_scores,
    )

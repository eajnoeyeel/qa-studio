"""Experiment endpoints (A/B and multi-comparison)."""
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...db.repository import ExperimentRepository
from ...models.schemas import ABExperimentRequest, ABExperimentResponse, MultiComparisonRequest, MultiComparisonSummary
from ...providers import get_provider
from ...services.experiment import ExperimentService
from ...services.instrumentation import LangfuseInstrumentation
from ...services.multi_compare import MultiCompareService
from ...services.pipeline import EvaluationPipeline
from ..deps import (
    SessionLocal,
    get_db,
    get_instrumentation,
    get_rag_retriever,
    settings,
    validate_model_version,
)

router = APIRouter()


@router.post("/experiment/ab", response_model=ABExperimentResponse)
async def run_ab_experiment(
    request: ABExperimentRequest,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Run A/B experiment."""
    validate_model_version(request.config_a.model_version)
    validate_model_version(request.config_b.model_version)

    retriever = get_rag_retriever()

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

    pipeline_a = EvaluationPipeline(
        provider=provider_a,
        retriever=retriever,
        instrumentation=instrumentation,
        db_session=db,
        session_factory=SessionLocal,
    )
    pipeline_b = EvaluationPipeline(
        provider=provider_b,
        retriever=retriever,
        instrumentation=instrumentation,
        db_session=db,
        session_factory=SessionLocal,
    )

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


@router.get("/experiments")
async def list_experiments(db: Session = Depends(get_db)):
    """List all experiments."""
    repo = ExperimentRepository(db)
    experiments = repo.get_all()
    return {"experiments": [e.model_dump() for e in experiments]}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Get experiment by ID."""
    from fastapi import HTTPException

    repo = ExperimentRepository(db)
    experiment = repo.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.model_dump()


@router.post("/experiment/multi", response_model=MultiComparisonSummary)
async def run_multi_comparison(
    request: MultiComparisonRequest,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Run N-way config comparison across a dataset split."""
    for cfg in request.configs:
        validate_model_version(cfg.model_version)

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
    from ...db.repository import MultiComparisonRepository

    repo = MultiComparisonRepository(db)
    results = repo.get_by_experiment(experiment_id)
    return {"experiment_id": experiment_id, "results": results}

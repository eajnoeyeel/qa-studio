"""Pattern analysis and suggestion endpoints."""
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...db.repository import FailurePatternRepository
from ...models.schemas import (
    DatasetSplit,
    FailurePattern,
    PatternAnalysisRequest,
    PatternAnalysisResult,
    PromptSuggestion,
    SuggestionGenerateRequest,
)
from ...providers import get_provider
from ...services.instrumentation import LangfuseInstrumentation
from ...services.pattern_analyzer import PatternAnalyzer
from ...services.prompt_suggester import PromptSuggester
from ..deps import get_db, get_instrumentation, settings

router = APIRouter()


@router.post("/analysis/patterns", response_model=PatternAnalysisResult)
async def run_pattern_analysis(
    request: PatternAnalysisRequest,
    db: Session = Depends(get_db),
):
    """Run failure pattern analysis across stored evaluations."""
    analyzer = PatternAnalyzer(db_session=db)
    return await analyzer.analyze(
        dataset_split=request.dataset_split,
        prompt_version=request.prompt_version,
        model_version=request.model_version,
        min_frequency=request.min_frequency,
        top_k=request.top_k,
    )


@router.get("/analysis/patterns/latest", response_model=List[FailurePattern])
async def get_latest_patterns(
    top_k: int = Query(10, ge=1, le=50),
    dataset_split: DatasetSplit | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get patterns from the most recent analysis run."""
    repo = FailurePatternRepository(db)
    return repo.get_latest(top_k=top_k, dataset_split=dataset_split)


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

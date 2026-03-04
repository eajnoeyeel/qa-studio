"""Composed API routes for QA Evaluation Studio."""
from fastapi import APIRouter

from ..services.pipeline import EvaluationPipeline
from .deps import (
    SessionLocal,
    build_rag_indexer,
    get_db,
    get_instrumentation,
    get_rag_retriever,
    set_rag_retriever,
    settings,
    validate_model_version,
)
from .endpoints import (
    analysis_router,
    documents_router,
    evaluate_router,
    experiments_router,
    health_router,
    human_router,
    ingest_router,
    items_router,
    prompts_router,
    proposals_router,
)

router = APIRouter()

router.include_router(ingest_router)
router.include_router(evaluate_router)
router.include_router(experiments_router)
router.include_router(items_router)
router.include_router(human_router)
router.include_router(documents_router)
router.include_router(health_router)
router.include_router(prompts_router)
router.include_router(analysis_router)
router.include_router(proposals_router)

__all__ = [
    "router",
    "settings",
    "SessionLocal",
    "get_db",
    "get_instrumentation",
    "validate_model_version",
    "get_rag_retriever",
    "set_rag_retriever",
    "build_rag_indexer",
    "EvaluationPipeline",
]

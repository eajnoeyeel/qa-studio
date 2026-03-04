"""Route modules."""

from .analysis import router as analysis_router
from .documents import router as documents_router
from .evaluate import router as evaluate_router
from .experiments import router as experiments_router
from .health import router as health_router
from .human import router as human_router
from .ingest import router as ingest_router
from .items import router as items_router
from .prompts import router as prompts_router
from .proposals import router as proposals_router

__all__ = [
    "ingest_router",
    "evaluate_router",
    "experiments_router",
    "items_router",
    "human_router",
    "documents_router",
    "health_router",
    "prompts_router",
    "analysis_router",
    "proposals_router",
]

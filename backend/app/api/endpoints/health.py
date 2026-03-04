"""Health endpoint."""
from datetime import datetime

from fastapi import APIRouter

from ..deps import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "langfuse_enabled": bool(settings.LANGFUSE_PUBLIC_KEY),
        "llm_provider": settings.LLM_PROVIDER,
    }

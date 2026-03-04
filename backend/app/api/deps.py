"""Shared API dependencies and helpers."""
from typing import Optional

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.database import init_db
from ..rag.indexer import RAGIndexer
from ..rag.retriever import RAGRetriever
from ..services.instrumentation import LangfuseInstrumentation

settings = get_settings()

# Initialize database once for API layer
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
        session_factory=SessionLocal,
    )



def validate_model_version(model_version: str) -> None:
    """Reject 'mock' model_version when using a real LLM provider."""
    if settings.LLM_PROVIDER != "mock" and model_version == "mock":
        raise HTTPException(
            status_code=422,
            detail=(
                f"model_version='mock' is invalid when LLM_PROVIDER='{settings.LLM_PROVIDER}'. "
                "Use a real model id (e.g. 'gpt-4o-mini')."
            ),
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
            openai_api_key=settings.OPENAI_API_KEY,
        )
        indexer.load_index() or indexer.build_index()
        _rag_retriever = RAGRetriever(indexer)
    return _rag_retriever



def set_rag_retriever(retriever: RAGRetriever) -> None:
    """Set cached RAG retriever instance."""
    global _rag_retriever
    _rag_retriever = retriever



def build_rag_indexer() -> RAGIndexer:
    """Build an indexer configured from settings."""
    use_mock = settings.LLM_PROVIDER == "mock" or not settings.OPENAI_API_KEY
    return RAGIndexer(
        docs_path=settings.DOCS_PATH,
        vector_store_path=settings.VECTOR_STORE_PATH,
        embedding_model=settings.EMBEDDING_MODEL,
        use_mock=use_mock,
        openai_api_key=settings.OPENAI_API_KEY,
    )

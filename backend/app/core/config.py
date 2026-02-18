"""Application configuration with environment variables."""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "SaaS CS QA Studio"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "sqlite:///./cs_qa_studio.db"

    # Langfuse (optional - graceful fallback if not set)
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # LLM Provider
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "mock"  # mock, openai, anthropic
    LLM_MODEL: str = "gpt-4o-mini"

    # RAG
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    RAG_TOP_K: int = 5
    DOCS_PATH: str = "./docs"
    VECTOR_STORE_PATH: str = "./vector_store"

    # Sampling thresholds
    AB_AMBIGUOUS_THRESHOLD: float = 2.0
    LOW_SCORE_THRESHOLD: int = 2

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

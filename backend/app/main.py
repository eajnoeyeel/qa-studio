"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .api.routes import router

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="QA Studio - Automated Q/A Response Quality Assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=settings.API_PREFIX)


def _run_migrations():
    """Apply lightweight schema migrations for new columns on existing DBs."""
    from sqlalchemy import create_engine, inspect, text
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    )
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("eval_items")}
    migrations = [
        ("scenario_id", "VARCHAR"),
        ("candidate_source", "VARCHAR"),
    ]
    with engine.begin() as conn:
        for col_name, col_type in migrations:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE eval_items ADD COLUMN {col_name} {col_type}"))
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS ix_eval_items_{col_name} ON eval_items({col_name})"
                ))
    engine.dispose()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    from .rag.indexer import RAGIndexer

    # Apply any pending schema migrations
    _run_migrations()

    # Build RAG index if not exists
    use_mock = settings.LLM_PROVIDER == "mock" or not settings.OPENAI_API_KEY
    indexer = RAGIndexer(
        docs_path=settings.DOCS_PATH,
        vector_store_path=settings.VECTOR_STORE_PATH,
        use_mock=use_mock,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    if not indexer.load_index():
        indexer.build_index()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "api": settings.API_PREFIX,
    }

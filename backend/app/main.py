"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .api.routes import router

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="SaaS CS QA Studio - Automated Customer Service Quality Assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    from .rag.indexer import RAGIndexer

    # Build RAG index if not exists
    use_mock = settings.LLM_PROVIDER == "mock" or not settings.OPENAI_API_KEY
    indexer = RAGIndexer(
        docs_path=settings.DOCS_PATH,
        vector_store_path=settings.VECTOR_STORE_PATH,
        use_mock=use_mock,
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

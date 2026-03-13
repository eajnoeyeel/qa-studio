"""FastAPI application entry point."""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

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


def _migrate_unique_index(conn, inspector, table, index_name, columns, is_sqlite):
    """Add a unique index, deduplicating existing rows first if needed.

    FK-safe: deletes dependents of duplicate rows before removing them.
    """
    from sqlalchemy import text

    existing_indexes = {idx["name"] for idx in inspector.get_indexes(table)}
    existing_uniques = {uc["name"] for uc in inspector.get_unique_constraints(table)}
    if index_name in existing_indexes or index_name in existing_uniques:
        return

    cols_csv = ", ".join(columns)

    # Build a subquery that identifies the duplicate IDs to remove
    # (keep the row with the smallest id per group)
    dupes_subquery = (
        f"SELECT id FROM {table} "
        f"WHERE id NOT IN (SELECT MIN(id) FROM {table} GROUP BY {cols_csv})"
    )

    # Check if duplicates actually exist before doing destructive work
    row = conn.execute(text(f"SELECT COUNT(*) FROM ({dupes_subquery}) t")).scalar()
    if row and row > 0:
        logger.warning(
            "Found %d duplicate rows in %s for (%s); removing them and their dependents.",
            row, table, cols_csv,
        )

        # Delete dependents that FK-reference the duplicate rows.
        # Order matters: human_reviews references both evaluations.id
        # and human_queue.id, so it must be deleted before human_queue.
        fk_children = {
            "evaluations": [
                ("human_reviews", "evaluation_id"),
                ("judge_outputs", "evaluation_id"),
                ("human_queue", "evaluation_id"),
                ("experiment_results", "eval_a_id"),
                ("experiment_results", "eval_b_id"),
            ],
        }
        for child_table, fk_col in fk_children.get(table, []):
            try:
                conn.execute(text(
                    f"DELETE FROM {child_table} "
                    f"WHERE {fk_col} IN ({dupes_subquery})"
                ))
            except Exception as e:
                logger.warning("Could not clean %s dependents: %s", child_table, e)

        # Now delete the duplicate rows themselves
        try:
            conn.execute(text(
                f"DELETE FROM {table} WHERE id IN ({dupes_subquery})"
            ))
        except Exception as e:
            logger.warning(
                "Could not deduplicate %s before adding unique index %s: %s",
                table, index_name, e,
            )
            return

    try:
        conn.execute(text(
            f"CREATE UNIQUE INDEX {index_name} ON {table}({cols_csv})"
        ))
        logger.info("Added unique index %s on %s(%s)", index_name, table, cols_csv)
    except Exception as e:
        logger.warning(
            "Failed to create unique index %s on %s: %s. "
            "Constraint is enforced on new databases via model definition.",
            index_name, table, e,
        )


def _run_migrations():
    """Apply lightweight schema migrations for new columns on existing DBs."""
    from sqlalchemy import create_engine, inspect, text
    is_sqlite = "sqlite" in settings.DATABASE_URL
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False, "timeout": 60} if is_sqlite else {},
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

        if inspector.has_table("evaluations"):
            evaluation_cols = {c["name"] for c in inspector.get_columns("evaluations")}
            evaluation_migrations = [
                ("evaluation_kind", "VARCHAR NOT NULL DEFAULT 'dataset'"),
                ("evaluated_question", "TEXT"),
                ("evaluated_response", "TEXT"),
                ("evaluated_system_prompt", "TEXT"),
            ]
            for col_name, col_type in evaluation_migrations:
                if col_name not in evaluation_cols:
                    conn.execute(text(f"ALTER TABLE evaluations ADD COLUMN {col_name} {col_type}"))
            if "evaluation_kind" not in evaluation_cols:
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_evaluations_evaluation_kind "
                    "ON evaluations(evaluation_kind)"
                ))

        if inspector.has_table("failure_patterns"):
            pattern_cols = {c["name"] for c in inspector.get_columns("failure_patterns")}
            if "dataset_split" not in pattern_cols:
                conn.execute(text("ALTER TABLE failure_patterns ADD COLUMN dataset_split VARCHAR"))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_failure_patterns_dataset_split "
                    "ON failure_patterns(dataset_split)"
                ))

        # Add prompt_type column to prompt_proposals if missing
        if inspector.has_table("prompt_proposals"):
            proposal_cols = {c["name"] for c in inspector.get_columns("prompt_proposals")}
            if "prompt_type" not in proposal_cols:
                conn.execute(text(
                    "ALTER TABLE prompt_proposals ADD COLUMN prompt_type VARCHAR NOT NULL DEFAULT 'system_prompt'"
                ))

        # Add unique constraints if missing
        _migrate_unique_index(
            conn, inspector, "evaluations",
            "uq_evaluation_version_triple",
            ["item_id", "prompt_version", "model_version", "docs_version"],
            is_sqlite=is_sqlite,
        )
        _migrate_unique_index(
            conn, inspector, "human_reviews",
            "uq_human_reviews_queue_item_id",
            ["queue_item_id"],
            is_sqlite=is_sqlite,
        )

    engine.dispose()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Apply any pending schema migrations
    _run_migrations()

    # RAG index is built lazily on first request via deps.get_rag_retriever()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "api": settings.API_PREFIX,
    }

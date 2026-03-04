#!/usr/bin/env python3
"""One-time migration: copy all tables from SQLite to PostgreSQL."""
import argparse
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent to path so we can import app models
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from app.models.database import Base

# Tables in FK-safe insertion order
TABLES = [
    "eval_items",
    "documents",
    "experiments",
    "traces",
    "failure_patterns",
    "prompt_proposals",
    "evaluations",
    "judge_outputs",
    "human_queue",
    "human_reviews",
    "experiment_results",
    "multi_comparison_results",
]


def migrate(sqlite_url: str, pg_url: str, batch_size: int):
    sqlite_engine = create_engine(sqlite_url)
    pg_engine = create_engine(pg_url)

    # Create tables in Postgres
    Base.metadata.create_all(bind=pg_engine)

    sqlite_session = sessionmaker(bind=sqlite_engine)()
    pg_session = sessionmaker(bind=pg_engine)()

    for table_name in TABLES:
        rows = sqlite_session.execute(text(f"SELECT * FROM {table_name}")).fetchall()
        if not rows:
            print(f"  {table_name}: 0 rows (skip)")
            continue

        columns = list(sqlite_session.execute(text(f"SELECT * FROM {table_name} LIMIT 1")).keys())
        col_list = ", ".join(columns)
        param_list = ", ".join(f":{c}" for c in columns)
        insert_sql = text(f"INSERT INTO {table_name} ({col_list}) VALUES ({param_list})")

        for i in range(0, len(rows), batch_size):
            batch = [dict(zip(columns, row)) for row in rows[i : i + batch_size]]
            pg_session.execute(insert_sql, batch)
            pg_session.commit()

        pg_count = pg_session.execute(text(f"SELECT count(*) FROM {table_name}")).scalar()
        print(f"  {table_name}: {len(rows)} -> {pg_count}")

    sqlite_session.close()
    pg_session.close()
    print("Migration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate QA Studio data from SQLite to PostgreSQL")
    parser.add_argument("--sqlite-url", default="sqlite:///./qa_studio.db")
    parser.add_argument("--pg-url", default="postgresql://qa_studio:qa_studio@localhost:5433/qa_studio")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    migrate(args.sqlite_url, args.pg_url, args.batch_size)

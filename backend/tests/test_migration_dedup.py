"""Test that _migrate_unique_index safely deduplicates rows with FK dependents."""
import uuid

import pytest
from sqlalchemy import create_engine, event, inspect, text


def _id():
    return uuid.uuid4().hex[:12]


@pytest.fixture()
def fresh_engine(tmp_path):
    """Isolated SQLite DB with OLD schema (no unique constraints on evaluations/human_reviews)."""
    db_path = tmp_path / "migration_test.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    # Create tables with raw SQL to simulate old schema (no unique constraints)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE eval_items (
                id VARCHAR PRIMARY KEY,
                question TEXT NOT NULL,
                response TEXT NOT NULL,
                split VARCHAR DEFAULT 'dev',
                system_prompt TEXT,
                metadata_json TEXT,
                scenario_id VARCHAR,
                candidate_source VARCHAR,
                masked_text TEXT,
                external_id VARCHAR,
                created_at DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE evaluations (
                id VARCHAR PRIMARY KEY,
                item_id VARCHAR NOT NULL REFERENCES eval_items(id),
                prompt_version VARCHAR NOT NULL,
                model_version VARCHAR NOT NULL,
                docs_version VARCHAR NOT NULL,
                classification_json TEXT,
                trace_id VARCHAR,
                created_at DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE judge_outputs (
                id VARCHAR PRIMARY KEY,
                evaluation_id VARCHAR NOT NULL UNIQUE REFERENCES evaluations(id),
                gates_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                failure_tags_json TEXT NOT NULL,
                summary_of_issue TEXT NOT NULL,
                what_to_fix TEXT NOT NULL,
                rag_citations_json TEXT,
                created_at DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE human_queue (
                id VARCHAR PRIMARY KEY,
                item_id VARCHAR NOT NULL REFERENCES eval_items(id),
                evaluation_id VARCHAR NOT NULL REFERENCES evaluations(id),
                reason VARCHAR NOT NULL,
                priority INTEGER DEFAULT 0,
                created_at DATETIME,
                reviewed BOOLEAN DEFAULT 0
            )
        """))
        conn.execute(text("""
            CREATE TABLE human_reviews (
                id VARCHAR PRIMARY KEY,
                queue_item_id VARCHAR NOT NULL REFERENCES human_queue(id),
                evaluation_id VARCHAR NOT NULL REFERENCES evaluations(id),
                reviewer_id VARCHAR,
                gold_label VARCHAR,
                gold_gates_json TEXT,
                gold_scores_json TEXT,
                gold_tags_json TEXT,
                notes TEXT,
                created_at DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE experiment_results (
                id VARCHAR PRIMARY KEY,
                experiment_id VARCHAR NOT NULL,
                item_id VARCHAR NOT NULL REFERENCES eval_items(id),
                eval_a_id VARCHAR NOT NULL REFERENCES evaluations(id),
                eval_b_id VARCHAR NOT NULL REFERENCES evaluations(id),
                score_diff_json TEXT NOT NULL,
                gate_diff_json TEXT NOT NULL,
                is_ambiguous BOOLEAN DEFAULT 0,
                winner VARCHAR
            )
        """))

    yield engine
    engine.dispose()


def _seed_duplicates(engine):
    """Insert two duplicate evaluations with full FK chains (judge, queue, review)."""
    item_id = _id()
    eval_id_keep = "aaa_" + _id()  # MIN(id) → kept
    eval_id_dup = "zzz_" + _id()   # larger id → duplicate
    queue_id_keep = _id()
    queue_id_dup = _id()

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO eval_items (id, question, response, split) "
            "VALUES (:id, 'q', 'r', 'dev')"
        ), {"id": item_id})

        for eid in (eval_id_keep, eval_id_dup):
            conn.execute(text(
                "INSERT INTO evaluations (id, item_id, prompt_version, model_version, docs_version) "
                "VALUES (:id, :item, 'v1', 'mock', 'v1')"
            ), {"id": eid, "item": item_id})

        for eid in (eval_id_keep, eval_id_dup):
            conn.execute(text(
                "INSERT INTO judge_outputs "
                "(id, evaluation_id, gates_json, scores_json, failure_tags_json, summary_of_issue, what_to_fix) "
                "VALUES (:id, :eid, '[]', '[]', '[]', '', '')"
            ), {"id": _id(), "eid": eid})

        for qid, eid in ((queue_id_keep, eval_id_keep), (queue_id_dup, eval_id_dup)):
            conn.execute(text(
                "INSERT INTO human_queue (id, item_id, evaluation_id, reason, priority) "
                "VALUES (:id, :item, :eid, 'gate_fail', 0)"
            ), {"id": qid, "item": item_id, "eid": eid})

        for qid, eid in ((queue_id_keep, eval_id_keep), (queue_id_dup, eval_id_dup)):
            conn.execute(text(
                "INSERT INTO human_reviews (id, queue_item_id, evaluation_id) "
                "VALUES (:id, :qid, :eid)"
            ), {"id": _id(), "qid": qid, "eid": eid})

    return item_id, eval_id_keep, eval_id_dup


def test_migrate_deduplicates_with_fk_chain(fresh_engine):
    """Migration should remove duplicates and all their FK dependents without errors."""
    from app.main import _migrate_unique_index

    item_id, eval_keep, eval_dup = _seed_duplicates(fresh_engine)

    inspector = inspect(fresh_engine)
    with fresh_engine.begin() as conn:
        _migrate_unique_index(
            conn, inspector, "evaluations",
            "uq_evaluation_version_triple",
            ["item_id", "prompt_version", "model_version", "docs_version"],
            is_sqlite=True,
        )

    with fresh_engine.connect() as conn:
        # Exactly one evaluation remains
        eval_count = conn.execute(text(
            "SELECT COUNT(*) FROM evaluations WHERE item_id = :item"
        ), {"item": item_id}).scalar()
        assert eval_count == 1

        remaining_id = conn.execute(text(
            "SELECT id FROM evaluations WHERE item_id = :item"
        ), {"item": item_id}).scalar()
        assert remaining_id == eval_keep  # MIN(id) kept

        # Surviving eval's dependents remain
        assert conn.execute(text(
            "SELECT COUNT(*) FROM judge_outputs WHERE evaluation_id = :eid"
        ), {"eid": remaining_id}).scalar() == 1

        # No orphans from deleted eval
        assert conn.execute(text(
            "SELECT COUNT(*) FROM human_reviews WHERE evaluation_id = :eid"
        ), {"eid": eval_dup}).scalar() == 0
        assert conn.execute(text(
            "SELECT COUNT(*) FROM human_queue WHERE evaluation_id = :eid"
        ), {"eid": eval_dup}).scalar() == 0
        assert conn.execute(text(
            "SELECT COUNT(*) FROM judge_outputs WHERE evaluation_id = :eid"
        ), {"eid": eval_dup}).scalar() == 0

    # Unique index now exists
    idx_names = {idx["name"] for idx in inspect(fresh_engine).get_indexes("evaluations")}
    assert "uq_evaluation_version_triple" in idx_names


def test_migrate_noop_when_no_duplicates(fresh_engine):
    """Migration should succeed silently when there are no duplicates."""
    from app.main import _migrate_unique_index

    inspector = inspect(fresh_engine)
    with fresh_engine.begin() as conn:
        _migrate_unique_index(
            conn, inspector, "evaluations",
            "uq_evaluation_version_triple",
            ["item_id", "prompt_version", "model_version", "docs_version"],
            is_sqlite=True,
        )

    idx_names = {idx["name"] for idx in inspect(fresh_engine).get_indexes("evaluations")}
    assert "uq_evaluation_version_triple" in idx_names


def test_migrate_human_reviews_unique(fresh_engine):
    """Migration should add unique index on human_reviews.queue_item_id."""
    from app.main import _migrate_unique_index

    inspector = inspect(fresh_engine)
    with fresh_engine.begin() as conn:
        _migrate_unique_index(
            conn, inspector, "human_reviews",
            "uq_human_reviews_queue_item_id",
            ["queue_item_id"],
            is_sqlite=True,
        )

    idx_names = {idx["name"] for idx in inspect(fresh_engine).get_indexes("human_reviews")}
    assert "uq_human_reviews_queue_item_id" in idx_names

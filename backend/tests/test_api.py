"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api import routes as api_routes
from app.db.repository import EvalItemRepository, EvaluationRepository, HumanQueueRepository
from app.models.schemas import EvalItemCreate, EvaluationCreate, HumanQueueReason


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "langfuse_enabled" in data
    assert "llm_provider" in data


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "QA Studio"
    assert "version" in data


def test_list_items_empty(client):
    """Test listing items when empty."""
    response = client.get("/api/v1/items")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)


def test_list_items_with_pagination(client):
    """Test listing items with pagination params."""
    response = client.get("/api/v1/items?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 10


def test_get_nonexistent_item(client):
    """Test getting an item that doesn't exist."""
    response = client.get("/api/v1/items/nonexistent-id")
    assert response.status_code == 404


def test_get_human_queue(client):
    """Test getting human review queue."""
    response = client.get("/api/v1/human/queue")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_experiments(client):
    """Test listing experiments."""
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data


def test_ingest_without_file_or_path(client):
    """Test ingest endpoint without file or path."""
    response = client.post(
        "/api/v1/ingest/batch",
        json={"split": "dev"}
    )
    # Should return 422 because file_path is a required field
    assert response.status_code == 422


def test_evaluate_run_validation(client):
    """Test evaluate run with valid request."""
    response = client.post(
        "/api/v1/evaluate/run",
        json={
            "dataset_split": "dev",
            "prompt_version": "v1",
            "model_version": "mock",
            "docs_version": "v1"
        }
    )
    # Should succeed even with no items
    assert response.status_code == 200
    data = response.json()
    assert "processed_count" in data
    assert "gate_fail_count" in data


def test_report_summary(client):
    """Test report summary endpoint."""
    response = client.get("/api/v1/reports/summary?dataset_split=dev")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_split" in data
    assert "total_evaluations" in data
    assert "gate_fail_rate" in data


def test_documents_reindex(client):
    """Test document reindex endpoint."""
    response = client.post("/api/v1/documents/reindex")
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "document_count" in data


def test_submit_human_review_returns_typed_gold_fields(client):
    """Ensure human review endpoint returns JSON-typed gold fields, not serialized strings."""
    db = api_routes.SessionLocal()
    try:
        item_repo = EvalItemRepository(db)
        eval_repo = EvaluationRepository(db)
        queue_repo = HumanQueueRepository(db)

        item = item_repo.create(
            EvalItemCreate(
                external_id="review-test-item",
                split="dev",
                question="What is 2+2?",
                response="4",
            ),
            commit=True,
        )
        evaluation = eval_repo.create(
            EvaluationCreate(
                item_id=item.id,
                prompt_version="v1",
                model_version="mock",
                docs_version="v1",
            ),
            commit=True,
        )
        queue_item = queue_repo.create(
            item_id=item.id,
            evaluation_id=evaluation.id,
            reason=HumanQueueReason.MANUAL,
            priority=1,
            commit=True,
        )
    finally:
        db.close()

    response = client.post(
        "/api/v1/human/review",
        json={
            "queue_item_id": queue_item.id,
            "evaluation_id": evaluation.id,
            "gold_label": "math",
            "gold_gates": {"hallucination": True},
            "gold_scores": {"clarity": 4},
            "gold_tags": ["off_topic"],
            "notes": "typed-fields-check",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["gold_gates"] == {"hallucination": True}
    assert data["gold_scores"] == {"clarity": 4}
    assert data["gold_tags"] == ["off_topic"]

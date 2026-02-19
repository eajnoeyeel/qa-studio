"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


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
    assert data["name"] == "SaaS CS QA Studio"
    assert "version" in data


def test_list_tickets_empty(client):
    """Test listing tickets when empty."""
    response = client.get("/api/v1/tickets")
    assert response.status_code == 200
    data = response.json()
    assert "tickets" in data
    assert "total" in data
    assert isinstance(data["tickets"], list)


def test_list_tickets_with_pagination(client):
    """Test listing tickets with pagination params."""
    response = client.get("/api/v1/tickets?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 10


def test_get_nonexistent_ticket(client):
    """Test getting a ticket that doesn't exist."""
    response = client.get("/api/v1/tickets/nonexistent-id")
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
    # Should succeed even with no tickets
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

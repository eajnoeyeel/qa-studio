"""Tests for RAG retriever."""
import pytest
from app.rag.indexer import RAGIndexer, DocumentLoader
from app.rag.retriever import RAGRetriever


@pytest.fixture
def mock_indexer(tmp_path):
    """Create mock indexer with test documents."""
    # Create test docs directory
    docs_dir = tmp_path / "docs"
    policies_dir = docs_dir / "policies"
    help_dir = docs_dir / "help_center"
    policies_dir.mkdir(parents=True)
    help_dir.mkdir(parents=True)

    # Create test documents
    (policies_dir / "test_billing.md").write_text("""---
doc_id: test_billing
title: Billing Policy
version: v1
tags: [billing_seats, billing_refund]
---

# Billing Policy

Seats can be added or removed at any time.
Prorated charges apply for mid-cycle changes.
Refunds are available within 14 days of purchase.
""")

    (help_dir / "test_sso.md").write_text("""---
doc_id: test_sso
title: SSO Configuration Guide
version: v1
tags: [login_sso]
source_url: https://help.example.com/sso
---

# SSO Configuration

SAML SSO is available on Business and Enterprise plans.
Common error codes:
- SAML_001: Invalid certificate
- SAML_002: User not provisioned
- SAML_003: Clock skew error

Contact your IdP administrator for configuration help.
""")

    # Create indexer
    indexer = RAGIndexer(
        docs_path=str(docs_dir),
        vector_store_path=str(tmp_path / "vector_store"),
        use_mock=True
    )
    indexer.build_index()

    return indexer


@pytest.fixture
def retriever(mock_indexer):
    """Create retriever with mock indexer."""
    return RAGRetriever(mock_indexer)


def test_retrieve_billing_query(retriever):
    """Test retrieval for billing-related query."""
    result = retriever.retrieve("How do I add more seats to my plan?", top_k=5)

    assert len(result.documents) > 0
    # Should return billing doc
    doc_ids = [d.doc_id for d in result.documents]
    assert "test_billing" in doc_ids


def test_retrieve_sso_query(retriever):
    """Test retrieval for SSO-related query."""
    result = retriever.retrieve("SAML_002 error when logging in", top_k=5)

    assert len(result.documents) > 0
    # Should return SSO doc
    doc_ids = [d.doc_id for d in result.documents]
    assert "test_sso" in doc_ids


def test_retrieve_with_tag_filter(retriever):
    """Test retrieval with tag filter."""
    result = retriever.retrieve(
        "refund policy",
        top_k=5,
        filter_tags=["billing_refund"]
    )

    # Should only return billing doc
    for doc in result.documents:
        assert "billing" in doc.doc_id.lower() or "billing_refund" in doc.tags


def test_check_claim_supported(retriever):
    """Test claim checking for supported claim."""
    result = retriever.check_claim("Seats can be added at any time")

    # Should find support in billing doc
    assert result["supported"] in [True, None]  # True or uncertain
    assert len(result["citations"]) > 0


def test_check_claim_unsupported(retriever):
    """Test claim checking for unsupported claim."""
    result = retriever.check_claim("We offer free unlimited storage forever")

    # Should not find strong support
    assert result["confidence"] < 0.7


def test_get_context_for_evaluation(retriever):
    """Test getting context for evaluation."""
    result = retriever.get_context_for_evaluation(
        question="How do I add more seats to my plan?",
        response="You can add seats in Settings > Billing",
        taxonomy_label="billing_seats",
        top_k=3
    )

    assert result.documents is not None
    assert len(result.documents) <= 3


class TestDocumentLoader:
    """Tests for document loader."""

    def test_load_markdown_with_frontmatter(self, tmp_path):
        """Test loading markdown with frontmatter."""
        docs_dir = tmp_path / "docs" / "policies"
        docs_dir.mkdir(parents=True)

        (docs_dir / "test.md").write_text("""---
doc_id: my_doc
title: My Document
version: v2
tags: [tag1, tag2]
---

# Content

This is the document content.
""")

        loader = DocumentLoader(str(tmp_path / "docs"))
        docs = loader.load_all()

        assert len(docs) == 1
        doc = docs[0]
        assert doc["doc_id"] == "my_doc"
        assert doc["title"] == "My Document"
        assert doc["version"] == "v2"
        assert "tag1" in doc["tags"]
        assert "Content" in doc["content"]

    def test_load_markdown_without_frontmatter(self, tmp_path):
        """Test loading markdown without frontmatter."""
        docs_dir = tmp_path / "docs" / "help_center"
        docs_dir.mkdir(parents=True)

        (docs_dir / "simple.md").write_text("# Simple Document\n\nJust content.")

        loader = DocumentLoader(str(tmp_path / "docs"))
        docs = loader.load_all()

        assert len(docs) == 1
        doc = docs[0]
        assert doc["doc_id"] == "simple"  # From filename
        assert doc["version"] == "v1"  # Default

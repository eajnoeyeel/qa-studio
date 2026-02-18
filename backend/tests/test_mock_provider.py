"""Tests for MockProvider."""
import pytest
from app.providers.mock import MockProvider


@pytest.fixture
def provider():
    """Create mock provider."""
    return MockProvider(seed=42)


@pytest.mark.asyncio
async def test_classify_billing_seats(provider):
    """Test classification of billing seats ticket."""
    text = "I need to add 5 more seats to our Team plan. We're onboarding new team members."

    result = await provider.classify(
        text,
        labels=["billing_seats", "billing_refund", "login_sso", "bug_report"]
    )

    assert result["label"] == "billing_seats"
    assert result["confidence"] > 0
    assert "seat_count" in result["required_slots"] or "current_plan" in result["required_slots"]


@pytest.mark.asyncio
async def test_classify_sso_login(provider):
    """Test classification of SSO login ticket."""
    text = "I keep getting SAML_002 error when trying to log in with SSO through Okta."

    result = await provider.classify(
        text,
        labels=["billing_seats", "login_sso", "permission_sharing", "bug_report"]
    )

    assert result["label"] == "login_sso"
    assert "idp_provider" in result["detected_slots"] or "error_code" in result["detected_slots"]


@pytest.mark.asyncio
async def test_evaluate_good_response(provider):
    """Test evaluation of a good response."""
    conversation = "[USER]: I need to add 5 more seats to our plan."
    candidate = """Happy to help you add seats!
    Here's what you need to do:
    1. Go to Settings → Billing
    2. Click 'Manage Seats'
    3. Enter the number of seats you want to add
    The charge will be prorated. Would you like me to walk through it step by step?
    """

    result = await provider.evaluate(conversation, candidate, rubric={})

    # Check gates
    assert len(result["gates"]) == 2
    policy_gate = next(g for g in result["gates"] if g["gate_type"] == "policy_safety")
    assert policy_gate["passed"] is True

    # Check scores
    assert len(result["scores"]) == 4
    actionability = next(s for s in result["scores"] if s["score_type"] == "actionability")
    assert actionability["score"] >= 3  # Should be good with step-by-step


@pytest.mark.asyncio
async def test_evaluate_pii_violation(provider):
    """Test that PII violations are caught."""
    conversation = "[USER]: I need help with my account."
    candidate = "Sure! First, could you send me your credit card number so I can verify your account?"

    result = await provider.evaluate(conversation, candidate, rubric={})

    # Policy safety gate should fail
    policy_gate = next(g for g in result["gates"] if g["gate_type"] == "policy_safety")
    assert policy_gate["passed"] is False
    assert "policy_pii" in result["failure_tags"]


@pytest.mark.asyncio
async def test_evaluate_overclaim(provider):
    """Test that overclaims are caught."""
    conversation = "[USER]: When will you add dark mode to mobile?"
    candidate = "Dark mode for mobile will definitely be released next month! I guarantee it."

    result = await provider.evaluate(conversation, candidate, rubric={})

    # Overclaim gate should fail
    overclaim_gate = next(g for g in result["gates"] if g["gate_type"] == "overclaim")
    assert overclaim_gate["passed"] is False
    assert "overclaim" in result["failure_tags"]


@pytest.mark.asyncio
async def test_complete_returns_response(provider):
    """Test that complete returns a mock response."""
    from app.providers.base import LLMMessage

    messages = [
        LLMMessage(role="user", content="Hello, how are you?")
    ]

    result = await provider.complete(messages)

    assert result.content is not None
    assert result.model == "mock-model"
    assert result.usage is not None

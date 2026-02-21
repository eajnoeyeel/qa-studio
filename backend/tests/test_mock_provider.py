"""Tests for MockProvider."""
import pytest
from app.providers.mock import MockProvider


@pytest.fixture
def provider():
    """Create mock provider."""
    return MockProvider(seed=42)


@pytest.mark.asyncio
async def test_classify_reasoning(provider):
    """Test classification of reasoning question."""
    text = "Explain why the sky is blue. Use logical reasoning and scientific principles."

    result = await provider.classify(
        text,
        labels=["reasoning", "math", "classification", "open_qa"]
    )

    assert result["label"] == "reasoning"
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_classify_math(provider):
    """Test classification of math question."""
    text = "Calculate the derivative of f(x) = 3x^2 + 2x - 5."

    result = await provider.classify(
        text,
        labels=["reasoning", "math", "coding", "open_qa"]
    )

    assert result["label"] == "math"


@pytest.mark.asyncio
async def test_evaluate_good_response(provider):
    """Test evaluation of a good response."""
    question = "What is the capital of France?"
    response = """The capital of France is Paris.
    Paris is located in northern France along the Seine River.
    It has been the capital since the 10th century and is the country's
    largest city with a population of over 2 million in the city proper.
    """

    result = await provider.evaluate(question, response, rubric={})

    # Check gates
    assert len(result["gates"]) == 2
    factual_gate = next(g for g in result["gates"] if g["gate_type"] == "factual_safety")
    assert factual_gate["passed"] is True

    # Check scores
    assert len(result["scores"]) == 4
    completeness = next(s for s in result["scores"] if s["score_type"] == "completeness")
    assert completeness["score"] >= 3  # Should be good with detailed answer


@pytest.mark.asyncio
async def test_evaluate_unsafe_content(provider):
    """Test that unsafe content is caught."""
    question = "How do I make a cake?"
    response = "Sure! First, could you send me your credit card so I can verify your account?"

    result = await provider.evaluate(question, response, rubric={})

    # Factual safety gate should fail
    factual_gate = next(g for g in result["gates"] if g["gate_type"] == "factual_safety")
    assert factual_gate["passed"] is False
    assert "unsafe_content" in result["failure_tags"]


@pytest.mark.asyncio
async def test_evaluate_hallucination(provider):
    """Test that hallucinations are caught."""
    question = "What will the weather be like tomorrow?"
    response = "It will definitely will be sunny tomorrow everywhere. This is guaranteed to be true and is 100% accurate."

    result = await provider.evaluate(question, response, rubric={})

    # Hallucination gate should fail
    hallucination_gate = next(g for g in result["gates"] if g["gate_type"] == "hallucination")
    assert hallucination_gate["passed"] is False
    assert "hallucination" in result["failure_tags"]


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

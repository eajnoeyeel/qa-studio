"""Tests for PromptSuggester suggestion generation and failure-tag coverage."""
import json
from datetime import datetime

import pytest

from app.models.schemas import (
    FailurePattern,
    PatternAnalysisResult,
    SuggestionGenerateRequest,
)
from app.providers.base import LLMMessage
from app.providers.mock import MockProvider
from app.services.prompt_suggester import PromptSuggester, _DEFAULT_PREAMBLE


def _make_pattern_result():
    pattern = FailurePattern(
        id="p-1",
        analysis_run_id="run-1",
        tags=["hallucination", "incomplete_answer"],
        frequency=10,
        avg_scores={"instruction_following": 2.0, "completeness": 2.0},
        taxonomy_labels={"open_qa": 5},
        created_at=datetime.utcnow(),
    )
    return PatternAnalysisResult(
        analysis_run_id="run-1",
        patterns_found=1,
        top_patterns=[pattern],
        total_evaluations_analyzed=100,
    )


@pytest.mark.asyncio
async def test_generate_suggestions_returns_prompt_with_tag_coverage(db_session):
    provider = MockProvider(seed=42)
    suggester = PromptSuggester(provider=provider, db_session=db_session)

    request = SuggestionGenerateRequest(prompt_name="system_prompt", top_k_patterns=5)
    suggestions = await suggester.generate_suggestions(request, _make_pattern_result())

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert "supported information" in suggestion.suggested_prompt.lower()
    assert "hallucination" in suggestion.coverage
    assert "incomplete_answer" in suggestion.coverage


@pytest.mark.asyncio
async def test_generate_suggestions_invalid_payload_falls_back_to_failure_tag_guidance(db_session):
    class BadPromptProvider(MockProvider):
        async def complete(self, messages, **kwargs):
            return type(
                "R",
                (),
                {"content": "not json", "model": "mock", "usage": {}},
            )()

    provider = BadPromptProvider(seed=42)
    suggester = PromptSuggester(provider=provider, db_session=db_session)

    request = SuggestionGenerateRequest(prompt_name="system_prompt", top_k_patterns=5)
    suggestions = await suggester.generate_suggestions(request, _make_pattern_result())

    assert len(suggestions) == 1
    prompt = suggestions[0].suggested_prompt.lower()
    assert prompt != _DEFAULT_PREAMBLE.lower()
    assert "information you can support" in prompt
    assert "address every part" in prompt


@pytest.mark.asyncio
async def test_generate_suggestions_empty_prompt_is_augmented_from_tag_guidance(db_session):
    class EmptyPromptProvider(MockProvider):
        async def complete(self, messages, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "system_prompt": "",
                            "rationale": "empty",
                            "expected_improvement": "none",
                            "coverage": {"hallucination": ""},
                        }
                    ),
                    "model": "mock",
                    "usage": {},
                },
            )()

    provider = EmptyPromptProvider(seed=42)
    suggester = PromptSuggester(provider=provider, db_session=db_session)

    request = SuggestionGenerateRequest(prompt_name="system_prompt", top_k_patterns=5)
    suggestions = await suggester.generate_suggestions(request, _make_pattern_result())

    assert len(suggestions) == 1
    prompt = suggestions[0].suggested_prompt.lower()
    assert "information you can support" in prompt
    assert "address every part" in prompt


@pytest.mark.asyncio
async def test_mock_complete_returns_system_prompt_and_coverage():
    provider = MockProvider(seed=42)
    msg = LLMMessage(role="user", content='Respond with JSON: {"system_prompt": "..."}')
    result = await provider.complete([msg])
    parsed = json.loads(result.content)

    assert "system_prompt" in parsed
    assert "coverage" in parsed
    assert isinstance(parsed["coverage"], dict)


def test_default_prompt_name_is_system_prompt():
    request = SuggestionGenerateRequest()
    assert request.prompt_name == "system_prompt"

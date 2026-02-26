"""Phase 4: Prompt Suggester — LLM-based prompt improvement suggestions."""
import uuid
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.schemas import PromptSuggestion, SuggestionGenerateRequest, PatternAnalysisResult
from ..providers.base import LLMProvider, LLMMessage

logger = logging.getLogger(__name__)

# Meta-prompt template for suggestion generation
META_PROMPT_TEMPLATE = """You are an expert prompt engineer. Your task is to improve an AI judge prompt
based on observed failure patterns in its evaluations.

Current Prompt:
{current_prompt}

Observed Failure Patterns (from {total_evaluations} evaluations):
{patterns_summary}

Based on these patterns, generate a specific, actionable improvement to the current prompt.
The improvement should directly address the most frequent failure patterns.

Respond with a JSON object:
{{
  "suggested_prompt": "The full improved prompt text",
  "rationale": "Why this change will help (2-3 sentences)",
  "expected_improvement": "Specific metric improvements expected (e.g., reduce hallucination tags by ~30%)"
}}

JSON Response:"""


class PromptSuggester:
    """Generates prompt improvement suggestions based on failure patterns."""
    _shared_suggestions: List[PromptSuggestion] = []

    def __init__(
        self,
        provider: LLMProvider,
        db_session: Session,
        instrumentation=None,
    ):
        self.provider = provider
        self.db = db_session
        self.instrumentation = instrumentation
        # Shared process-level store so "latest suggestions" survives per-request
        # service instantiation in the API layer.
        self._suggestions = self._shared_suggestions

    async def generate_suggestions(
        self,
        request: SuggestionGenerateRequest,
        pattern_result: Optional[PatternAnalysisResult] = None,
    ) -> List[PromptSuggestion]:
        """
        Analyze failure patterns and generate prompt improvement suggestions.

        Uses top_k_patterns from the latest analysis run to build a meta-prompt,
        then calls the LLM to produce targeted suggestions.
        """
        # Use provided pattern result or fetch latest from DB
        if pattern_result is None:
            pattern_result = self._get_latest_patterns(request.top_k_patterns)

        if not pattern_result or not pattern_result.top_patterns:
            logger.info("No failure patterns available for suggestion generation")
            return []

        # Get current prompt text
        current_prompt = self._get_current_prompt(request.prompt_name)

        # Build patterns summary for meta-prompt
        patterns_summary = self._build_patterns_summary(
            pattern_result.top_patterns[:request.top_k_patterns]
        )

        meta_prompt = META_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt,
            total_evaluations=pattern_result.total_evaluations_analyzed,
            patterns_summary=patterns_summary,
        )

        # Call LLM for suggestion
        try:
            response = await self.provider.complete(
                [LLMMessage(role="user", content=meta_prompt)],
                temperature=0.3,
                max_tokens=2000,
            )
            raw = response.content

            # Strip markdown code fences if present
            if raw.strip().startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines()
                    if not line.strip().startswith("```")
                )

            result = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse suggestion response: {e}")
            result = {
                "suggested_prompt": current_prompt,
                "rationale": "Suggestion generation failed — using current prompt.",
                "expected_improvement": "N/A",
            }

        # Build suggestion object
        target_patterns = [p.id for p in pattern_result.top_patterns[:request.top_k_patterns]]
        suggestion = PromptSuggestion(
            id=str(uuid.uuid4()),
            prompt_name=request.prompt_name,
            current_prompt_summary=current_prompt[:200] + "..." if len(current_prompt) > 200 else current_prompt,
            suggested_prompt=result.get("suggested_prompt", current_prompt),
            rationale=result.get("rationale", ""),
            target_patterns=target_patterns,
            expected_improvement=result.get("expected_improvement", ""),
            created_at=datetime.utcnow(),
        )

        # Optionally register in Langfuse as draft
        if request.register_in_langfuse and self.instrumentation:
            try:
                self.instrumentation.create_prompt(
                    name=f"{request.prompt_name}_suggestion",
                    prompt=suggestion.suggested_prompt,
                    labels=["draft"],
                )
                logger.info(f"Registered suggestion for '{request.prompt_name}' in Langfuse (draft)")
            except Exception as e:
                logger.warning(f"Failed to register suggestion in Langfuse: {e}")

        self._suggestions.append(suggestion)
        return [suggestion]

    def get_latest_suggestions(self, top_k: int = 5) -> List[PromptSuggestion]:
        """Return the most recent suggestions (in-memory)."""
        return self._suggestions[-top_k:]

    def _get_latest_patterns(self, top_k: int) -> Optional[PatternAnalysisResult]:
        """Fetch the latest pattern analysis result from DB."""
        try:
            from ..db.repository import FailurePatternRepository
            from ..models.database import EvaluationModel
            from sqlalchemy import func

            repo = FailurePatternRepository(self.db)
            patterns = repo.get_latest(top_k)
            if not patterns:
                return None
            run_id = repo.get_latest_run_id()

            # Get actual evaluation count so the meta-prompt is accurate
            total_evals = self.db.query(func.count(EvaluationModel.id)).scalar() or 0

            return PatternAnalysisResult(
                analysis_run_id=run_id or "unknown",
                patterns_found=len(patterns),
                top_patterns=patterns,
                total_evaluations_analyzed=total_evals,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch latest patterns: {e}")
            return None

    def _get_current_prompt(self, prompt_name: str) -> str:
        """Get current prompt text from Langfuse or return a default description."""
        if self.instrumentation:
            prompt_obj = self.instrumentation.get_prompt(prompt_name, label="production")
            if prompt_obj and hasattr(prompt_obj, "prompt"):
                return prompt_obj.prompt

        # Return a descriptive fallback
        return (
            f"[Hardcoded fallback for '{prompt_name}']: "
            "You are a QA evaluator. Evaluate the response for factual safety, "
            "hallucination, instruction following, reasoning quality, completeness, and clarity."
        )

    def _build_patterns_summary(self, patterns) -> str:
        """Build a human-readable summary of failure patterns for the meta-prompt."""
        lines = []
        for i, p in enumerate(patterns, 1):
            tags_str = " + ".join(p.tags)
            score_str = ", ".join(
                f"{k}: {v:.1f}" for k, v in p.avg_scores.items()
            ) if p.avg_scores else "N/A"
            taxonomy_str = ", ".join(
                f"{lbl}({cnt})" for lbl, cnt in sorted(
                    p.taxonomy_labels.items(), key=lambda x: -x[1]
                )[:3]
            ) if p.taxonomy_labels else "unknown"

            lines.append(
                f"{i}. Tags: [{tags_str}] | Frequency: {p.frequency} | "
                f"Avg Scores: {{{score_str}}} | Task Types: {taxonomy_str}"
            )
        return "\n".join(lines)

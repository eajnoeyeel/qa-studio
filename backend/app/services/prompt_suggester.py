"""Phase 4: Prompt Suggester — LLM-based system prompt improvement suggestions.

The self-improvement cycle improves the SYSTEM PROMPT (instructions given to the
response-generating LLM), NOT the judge/evaluation prompt. The judge is a fixed
measurement tool.
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ..models.database import EvalItemModel, EvaluationModel
from ..models.schemas import (
    EvaluationKind,
    PatternAnalysisResult,
    PromptSuggestion,
    SuggestionGenerateRequest,
)
from ..providers.base import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_PREAMBLE = "You are a helpful AI assistant. Provide clear, accurate, and complete answers."
SYSTEM_PROMPT_TEMPLATE = _DEFAULT_PREAMBLE

DEFAULT_TAG_GUIDANCE: Dict[str, str] = {
    "hallucination": "Answer only from information you can support, and clearly state when the available information is insufficient.",
    "instruction_miss": "Follow every explicit instruction and requested format before adding optional detail.",
    "incomplete_answer": "Address every part of the user's request before finishing the answer.",
    "partial_answer": "Do not stop after a partial response when the request has multiple parts.",
    "logic_error": "Use clear step-by-step reasoning when the task requires explanation, comparison, or analysis.",
    "format_violation": "Match the requested format exactly.",
    "over_verbose": "Be concise and avoid unnecessary filler once the answer is complete.",
    "under_verbose": "Provide enough detail to fully answer the request.",
    "wrong_language": "Respond in the language requested by the user.",
    "unsafe_content": "Decline unsafe instructions and redirect to safe, useful help.",
    "citation_missing": "When claims depend on provided sources, reference the supporting information instead of inventing support.",
    "off_topic": "Stay focused on the user's actual question and avoid unrelated content.",
}

META_PROMPT_TEMPLATE = """You are an expert prompt engineer. Improve the SYSTEM PROMPT used by an AI assistant.

The judge/evaluation prompt is fixed. Do not change the judge. Improve only the system prompt that generates user-facing answers.

Current System Prompt:
{current_preamble}

Top Failure Patterns (from {total_evaluations} dataset evaluations):
{patterns_summary}

Representative Failed Cases:
{pattern_examples}

Top Failure Tags That Must Be Addressed:
{top_tags}

Write an improved system prompt that directly targets those response failures.
Every top failure tag must have a corresponding instruction in the coverage map.
The instructions must address the cause of failure, not the judge.

Respond with a JSON object:
{{
  "system_prompt": "The improved system prompt",
  "rationale": "Why this change should improve response quality",
  "expected_improvement": "Expected gate/score/tag improvements",
  "coverage": {{
    "tag_name": "Instruction in the system prompt that addresses this failure cause"
  }}
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
        self._suggestions = self._shared_suggestions

    async def generate_suggestions(
        self,
        request: SuggestionGenerateRequest,
        pattern_result: Optional[PatternAnalysisResult] = None,
    ) -> List[PromptSuggestion]:
        """Generate a system prompt proposal grounded in the top failure tags."""
        if pattern_result is None:
            pattern_result = self._get_latest_patterns(
                top_k=request.top_k_patterns,
                dataset_split=request.dataset_split,
            )

        if not pattern_result or not pattern_result.top_patterns:
            logger.info("No failure patterns available for suggestion generation")
            return []

        patterns = pattern_result.top_patterns[:request.top_k_patterns]
        current_prompt = self._get_production_template(request.prompt_name)
        top_tags = self._derive_top_tags(patterns)
        patterns_summary = self._build_patterns_summary(patterns)
        pattern_examples = self._build_pattern_examples(patterns, request.dataset_split)

        meta_prompt = META_PROMPT_TEMPLATE.format(
            current_preamble=current_prompt,
            total_evaluations=pattern_result.total_evaluations_analyzed,
            patterns_summary=patterns_summary,
            pattern_examples=pattern_examples,
            top_tags=", ".join(top_tags) if top_tags else "(none)",
        )

        result = await self._request_suggestion(meta_prompt)
        raw_prompt = result.get("system_prompt", "")
        coverage = self._merge_coverage(top_tags, result.get("coverage"))
        suggested_prompt = self._finalize_prompt(
            current_prompt=current_prompt,
            raw_prompt=raw_prompt,
            coverage=coverage,
            top_tags=top_tags,
        )

        if self._normalize_text(suggested_prompt) == self._normalize_text(current_prompt):
            logger.warning("Suggestion generation produced no prompt change; skipping proposal")
            return []

        suggestion = PromptSuggestion(
            id=str(uuid.uuid4()),
            prompt_name=request.prompt_name,
            current_prompt_summary=(
                current_prompt[:200] + "..." if len(current_prompt) > 200 else current_prompt
            ),
            suggested_prompt=suggested_prompt,
            rationale=result.get("rationale") or self._build_default_rationale(top_tags),
            target_patterns=[p.id for p in patterns],
            expected_improvement=(
                result.get("expected_improvement")
                or self._build_default_expected_improvement(top_tags)
            ),
            coverage=coverage,
            created_at=datetime.utcnow(),
        )

        if request.register_in_langfuse and self.instrumentation:
            try:
                self.instrumentation.create_prompt(
                    name=f"{request.prompt_name}_suggestion",
                    prompt=suggestion.suggested_prompt,
                    labels=["draft"],
                )
                logger.info(
                    "Registered suggestion for '%s' in Langfuse (draft)",
                    request.prompt_name,
                )
            except Exception as exc:
                logger.warning("Failed to register suggestion in Langfuse: %s", exc)

        self._suggestions.append(suggestion)
        return [suggestion]

    def get_latest_suggestions(self, top_k: int = 5) -> List[PromptSuggestion]:
        """Return the most recent suggestions (in-memory)."""
        return self._suggestions[-top_k:]

    async def _request_suggestion(self, meta_prompt: str) -> Dict[str, object]:
        """Call the LLM and parse the suggestion payload."""
        try:
            response = await self.provider.complete(
                [LLMMessage(role="user", content=meta_prompt)],
                temperature=0.3,
                max_tokens=2000,
            )
            raw = response.content
            if raw.strip().startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines()
                    if not line.strip().startswith("```")
                )
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.warning("Failed to parse suggestion response: %s", exc)
            return {}

    def _get_latest_patterns(
        self,
        top_k: int,
        dataset_split=None,
    ) -> Optional[PatternAnalysisResult]:
        """Fetch the latest pattern analysis result from DB."""
        try:
            from ..db.repository import FailurePatternRepository

            repo = FailurePatternRepository(self.db)
            patterns = repo.get_latest(top_k, dataset_split=dataset_split)
            if not patterns:
                return None
            run_id = repo.get_latest_run_id(dataset_split=dataset_split)

            total_evals_query = self.db.query(func.count(EvaluationModel.id)).filter(
                EvaluationModel.evaluation_kind == EvaluationKind.DATASET
            )
            if dataset_split is not None:
                total_evals_query = total_evals_query.join(
                    EvalItemModel,
                    EvalItemModel.id == EvaluationModel.item_id,
                ).filter(EvalItemModel.split == dataset_split)
            total_evals = total_evals_query.scalar() or 0

            return PatternAnalysisResult(
                analysis_run_id=run_id or "unknown",
                patterns_found=len(patterns),
                top_patterns=patterns,
                total_evaluations_analyzed=total_evals,
                dataset_split=dataset_split,
            )
        except Exception as exc:
            logger.warning("Failed to fetch latest patterns: %s", exc)
            return None

    def _get_production_template(self, prompt_name: str) -> str:
        """Get the current system prompt from Langfuse or hardcoded fallback."""
        if self.instrumentation:
            prompt_obj = self.instrumentation.get_prompt(prompt_name, label="production")
            if prompt_obj and hasattr(prompt_obj, "prompt"):
                return prompt_obj.prompt
        return SYSTEM_PROMPT_TEMPLATE

    def _derive_top_tags(self, patterns) -> List[str]:
        """Flatten the top patterns into a stable ordered tag list."""
        seen = set()
        tags: List[str] = []
        for pattern in patterns:
            for tag in pattern.tags:
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        return tags

    def _build_patterns_summary(self, patterns) -> str:
        """Build a human-readable summary of failure patterns for the meta-prompt."""
        lines = []
        for idx, pattern in enumerate(patterns, start=1):
            tags_str = " + ".join(pattern.tags)
            score_str = ", ".join(
                f"{key}: {value:.1f}" for key, value in pattern.avg_scores.items()
            ) if pattern.avg_scores else "N/A"
            taxonomy_str = ", ".join(
                f"{label}({count})"
                for label, count in sorted(
                    pattern.taxonomy_labels.items(),
                    key=lambda entry: -entry[1],
                )[:3]
            ) if pattern.taxonomy_labels else "unknown"
            lines.append(
                f"{idx}. Tags: [{tags_str}] | Frequency: {pattern.frequency} | "
                f"Avg Scores: {{{score_str}}} | Task Types: {taxonomy_str}"
            )
        return "\n".join(lines)

    def _build_pattern_examples(self, patterns, dataset_split=None) -> str:
        """Attach representative failed examples so prompt changes target real causes."""
        query = self.db.query(EvaluationModel).options(
            joinedload(EvaluationModel.item),
            joinedload(EvaluationModel.judge_output),
        ).filter(EvaluationModel.evaluation_kind == EvaluationKind.DATASET)
        if dataset_split is not None:
            query = query.join(
                EvalItemModel,
                EvalItemModel.id == EvaluationModel.item_id,
            ).filter(EvalItemModel.split == dataset_split)

        evaluations = query.order_by(EvaluationModel.created_at.desc()).limit(300).all()
        examples: List[str] = []
        for pattern in patterns:
            pattern_tags = set(pattern.tags)
            match = next(
                (
                    evaluation for evaluation in evaluations
                    if evaluation.judge_output
                    and pattern_tags.issubset(set(evaluation.judge_output.failure_tags))
                    and evaluation.item is not None
                ),
                None,
            )
            if match is None or match.item is None or match.judge_output is None:
                continue
            examples.append(
                f"Pattern [{', '.join(pattern.tags)}]\n"
                f"Question: {self._clip(match.item.question, 180)}\n"
                f"Response: {self._clip(match.evaluated_response or match.item.response, 220)}\n"
                f"Issue: {self._clip(match.judge_output.summary_of_issue, 160)}\n"
                f"What to fix: {self._clip(match.judge_output.what_to_fix, 180)}"
            )
        return "\n\n".join(examples) if examples else "No representative failed cases available."

    def _merge_coverage(self, top_tags: List[str], raw_coverage) -> Dict[str, str]:
        """Fill in missing tag guidance so every top tag gets an actionable directive."""
        parsed = raw_coverage if isinstance(raw_coverage, dict) else {}
        coverage: Dict[str, str] = {}
        for tag in top_tags:
            value = parsed.get(tag)
            if not isinstance(value, str) or not value.strip():
                value = DEFAULT_TAG_GUIDANCE.get(
                    tag,
                    "Add an explicit instruction that prevents this failure mode in future responses.",
                )
            coverage[tag] = self._ensure_sentence(value.strip())
        return coverage

    def _finalize_prompt(
        self,
        current_prompt: str,
        raw_prompt: object,
        coverage: Dict[str, str],
        top_tags: List[str],
    ) -> str:
        """Build a final candidate prompt that explicitly covers each top tag."""
        base_prompt = raw_prompt.strip() if isinstance(raw_prompt, str) and raw_prompt.strip() else current_prompt.strip()
        additions = [
            coverage[tag]
            for tag in top_tags
            if coverage[tag].lower() not in base_prompt.lower()
        ]
        parts = [base_prompt] + additions
        return " ".join(part.strip() for part in parts if part.strip())

    def _build_default_rationale(self, top_tags: List[str]) -> str:
        if not top_tags:
            return "The candidate prompt adds clearer response-quality instructions."
        return (
            "The candidate prompt adds explicit instructions for the main failure causes: "
            + ", ".join(top_tags)
            + "."
        )

    def _build_default_expected_improvement(self, top_tags: List[str]) -> str:
        if not top_tags:
            return "Improve average gate and score outcomes."
        return (
            "Reduce failures in "
            + ", ".join(top_tags)
            + " while improving instruction_following, completeness, and clarity."
        )

    def _clip(self, text: Optional[str], limit: int) -> str:
        value = (text or "").strip()
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    def _ensure_sentence(self, text: str) -> str:
        if not text:
            return text
        return text if text[-1] in ".!?" else text + "."

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip().lower()

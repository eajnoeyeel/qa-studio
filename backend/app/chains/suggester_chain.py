"""Phase 3/4: LCEL suggester chain for prompt improvement generation."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SuggesterChain:
    """
    LCEL chain for generating prompt improvement suggestions.

    Takes failure pattern data as input and produces a structured
    suggestion via the LLMProvider. Used by Phase 4 PromptSuggester.

    Usage:
        chain = SuggesterChain(provider)
        result = await chain.invoke({
            "current_prompt": "...",
            "patterns_summary": "...",
            "total_evaluations": 100,
        })
        # result: {"suggested_prompt": "...", "rationale": "...", "expected_improvement": "..."}
    """

    META_PROMPT = """You are an expert prompt engineer. Improve the AI judge prompt below
based on observed failure patterns.

Current Prompt:
{current_prompt}

Observed Failure Patterns (from {total_evaluations} evaluations):
{patterns_summary}

Generate a specific, actionable improvement. Respond with JSON only:
{{
  "suggested_prompt": "Full improved prompt text",
  "rationale": "Why this helps (2-3 sentences)",
  "expected_improvement": "Expected metric improvements"
}}"""

    def __init__(self, provider, instrumentation=None):
        self.provider = provider
        self.instrumentation = instrumentation
        self._chain = None

    def build(self):
        """Build and cache the LCEL chain."""
        try:
            from langchain_core.runnables import RunnableLambda
            from langchain_core.prompts import PromptTemplate
            from .base import safe_json_parse

            prompt_template = PromptTemplate.from_template(self.META_PROMPT)

            async def _suggest(inputs: Dict[str, Any]) -> Dict[str, Any]:
                from ..providers.base import LLMMessage
                prompt_text = prompt_template.format(**inputs)
                response = await self.provider.complete(
                    [LLMMessage(role="user", content=prompt_text)],
                    temperature=0.3,
                    max_tokens=2000,
                )
                return safe_json_parse(response.content, fallback={
                    "suggested_prompt": inputs.get("current_prompt", ""),
                    "rationale": "Parse error — returning current prompt.",
                    "expected_improvement": "N/A",
                })

            self._chain = RunnableLambda(_suggest)
            logger.debug("LCEL SuggesterChain built successfully")
        except ImportError:
            logger.debug("langchain_core not available; SuggesterChain will use direct calls")
            self._chain = None

        return self

    async def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the suggester chain."""
        from .base import safe_json_parse
        from ..providers.base import LLMMessage

        if self._chain is not None:
            try:
                if hasattr(self._chain, "ainvoke"):
                    return await self._chain.ainvoke(inputs)
                return self._chain.invoke(inputs)
            except Exception as e:
                logger.warning(f"SuggesterChain invocation failed, falling back: {e}")

        # Fallback: direct call
        prompt_text = self.META_PROMPT.format(**inputs)
        response = await self.provider.complete(
            [LLMMessage(role="user", content=prompt_text)],
            temperature=0.3,
            max_tokens=2000,
        )
        return safe_json_parse(response.content, fallback={
            "suggested_prompt": inputs.get("current_prompt", ""),
            "rationale": "Parse error — returning current prompt.",
            "expected_improvement": "N/A",
        })

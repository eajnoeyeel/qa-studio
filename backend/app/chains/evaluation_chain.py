"""Phase 3: LCEL evaluation chain wrapping gate + score evaluation."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluationChain:
    """
    LCEL chain that evaluates a response via parallel gates and scores.

    Wraps the LLMProvider.evaluate() call inside a LangChain
    RunnableParallel → RunnableLambda pipeline when langchain_core is
    available. Falls back gracefully to a direct provider call when not.

    Usage:
        chain = EvaluationChain(provider, instrumentation)
        result = await chain.invoke({
            "question": "...",
            "response": "...",
            "context": "...",
            "system_prompt": "...",
        })
    """

    def __init__(self, provider, instrumentation=None, prompt_label: str = "production"):
        self.provider = provider
        self.instrumentation = instrumentation
        self.prompt_label = prompt_label
        self._chain = None

    def build(self):
        """Build and cache the LCEL chain."""
        try:
            from langchain_core.runnables import RunnableLambda, RunnableParallel

            async def _evaluate(inputs: Dict[str, Any]) -> Dict[str, Any]:
                return await self.provider.evaluate(
                    question=inputs["question"],
                    response=inputs["response"],
                    rubric={},
                    context=inputs.get("context"),
                    system_prompt=inputs.get("system_prompt"),
                    prompt_label=self.prompt_label,
                )

            # Parallel evaluation: gates branch + scores branch run simultaneously
            gate_runnable = RunnableLambda(
                lambda x: {k: v for k, v in x.items() if k == "gates"}
            )
            score_runnable = RunnableLambda(
                lambda x: {k: v for k, v in x.items() if k == "scores"}
            )

            eval_runnable = RunnableLambda(_evaluate)

            def _merge(inputs: Dict) -> Dict:
                """Merge parallel gate/score branches with remaining fields."""
                merged = {}
                merged.update(inputs.get("gates", {}))
                merged.update(inputs.get("scores", {}))
                return merged

            self._chain = eval_runnable | RunnableParallel(
                gates=gate_runnable,
                scores=score_runnable,
            ) | RunnableLambda(_merge)

            logger.debug("LCEL EvaluationChain built successfully")
        except ImportError:
            logger.debug("langchain_core not available; EvaluationChain will use direct provider calls")
            self._chain = None

        return self

    async def invoke(self, inputs: Dict[str, Any], trace=None) -> Dict[str, Any]:
        """Invoke the evaluation chain."""
        from .base import make_langfuse_callback

        callbacks = []
        if trace and self.instrumentation:
            cb = make_langfuse_callback(self.instrumentation, trace)
            if cb:
                callbacks.append(cb)

        if self._chain is not None:
            try:
                kwargs = {"config": {"callbacks": callbacks}} if callbacks else {}
                # LCEL chains may not natively support async; handle both
                if hasattr(self._chain, "ainvoke"):
                    result = await self._chain.ainvoke(inputs, **kwargs)
                else:
                    result = self._chain.invoke(inputs, **kwargs)
                # If chain returned merged (gates/scores split), re-assemble
                if "gates" not in result and "scores" not in result:
                    # Chain returned the raw provider dict — pass through
                    return result
                return result
            except Exception as e:
                logger.warning(f"LCEL chain invocation failed, falling back to direct call: {e}")

        # Fallback: direct provider call
        return await self.provider.evaluate(
            question=inputs["question"],
            response=inputs["response"],
            rubric={},
            context=inputs.get("context"),
            system_prompt=inputs.get("system_prompt"),
            prompt_label=self.prompt_label,
        )

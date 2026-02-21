"""Base utilities for LangChain LCEL chains."""
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def make_langfuse_callback(instrumentation, trace):
    """
    Return a LangChain callback handler that forwards spans to Langfuse.

    Falls back to a no-op handler when Langfuse is disabled.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler

        class _LangfuseCallback(BaseCallbackHandler):
            def on_chain_start(self, serialized, inputs, **kwargs):
                self._start = time.time()
                self._name = (serialized or {}).get("id", ["chain"])[-1]

            def on_chain_end(self, outputs, **kwargs):
                latency = (time.time() - getattr(self, "_start", time.time())) * 1000
                if instrumentation:
                    instrumentation.record_span(
                        trace,
                        getattr(self, "_name", "chain"),
                        output_data={"status": "success"},
                        latency_ms=latency,
                    )

            def on_chain_error(self, error, **kwargs):
                latency = (time.time() - getattr(self, "_start", time.time())) * 1000
                if instrumentation:
                    instrumentation.record_span(
                        trace,
                        getattr(self, "_name", "chain"),
                        error=str(error),
                        latency_ms=latency,
                    )

        return _LangfuseCallback()

    except ImportError:
        logger.debug("langchain_core not available; using no-op callback")
        return None


def safe_json_parse(text: str, fallback: Any = None) -> Any:
    """Parse JSON, stripping markdown code fences if present."""
    import json
    if text.strip().startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        )
    try:
        return json.loads(text)
    except (json.JSONDecodeError, Exception):
        return fallback

"""Langfuse instrumentation wrapper with graceful fallback.

Compatible with Langfuse SDK v3.x.
"""
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class LangfuseInstrumentation:
    """Wrapper for Langfuse tracing with fallback to local logging."""
    _prompt_cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
    _missing_prompt_until: Dict[Tuple[str, str], float] = {}
    _prompt_cache_ttl_seconds: float = 300.0
    _missing_prompt_ttl_seconds: float = 60.0

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        db_session=None,
        session_factory=None,
    ):
        self.enabled = bool(public_key and secret_key)
        self.langfuse = None
        self.db_session = db_session
        self.session_factory = session_factory

        if self.enabled:
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse instrumentation enabled")
            except ImportError:
                logger.warning("Langfuse package not installed, using fallback")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Langfuse initialization failed: {e}")
                self.enabled = False

        if not self.enabled:
            logger.info("Using local trace logging (Langfuse disabled)")

    def create_trace(
        self,
        trace_id: str,
        name: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a Langfuse trace via a root span with trace-level metadata."""
        if self.enabled and self.langfuse:
            try:
                root_span = self.langfuse.start_span(
                    name=name,
                    metadata=metadata or {},
                )
                # Populate trace-level fields so the Langfuse UI shows
                # Name, Tags, and Metadata in the traces list view.
                root_span.update_trace(
                    name=name,
                    tags=tags or [],
                    metadata=metadata or {},
                )
                return LangfuseTraceWrapper(
                    trace_id=trace_id,
                    name=name,
                    root_span=root_span,
                    langfuse=self.langfuse,
                    tags=tags,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Failed to create Langfuse trace: {e}")

        # Fallback: return mock trace
        return MockTrace(trace_id, name, tags, metadata, self.db_session)

    @contextmanager
    def span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None
    ):
        """Context manager for spans."""
        start_time = time.time()
        error_msg = None

        try:
            yield
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000

            # Record span on the trace wrapper
            if hasattr(trace, 'record_child_span'):
                trace.record_child_span(name, input_data, latency_ms, error_msg)

            # Log to fallback
            self._log_span(trace, name, input_data, latency_ms, error_msg)

    def record_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Record a completed span."""
        if hasattr(trace, 'record_child_span'):
            trace.record_child_span(name, input_data, latency_ms, error, output_data)

        self._log_span(trace, name, input_data, latency_ms, error, output_data=output_data)

    def record_score(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """Record a score on the trace."""
        if hasattr(trace, 'score'):
            trace.score(name=name, value=value, comment=comment)
        elif self.enabled and self.langfuse:
            try:
                trace_id = getattr(trace, 'trace_id', None)
                self.langfuse.create_score(
                    name=name,
                    value=value,
                    trace_id=trace_id,
                    comment=comment,
                    data_type="NUMERIC",
                )
            except Exception as e:
                logger.error(f"Failed to record Langfuse score: {e}")

        logger.info(f"Score recorded: {name}={value} ({comment})")

    def _log_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]],
        latency_ms: Optional[float],
        error: Optional[str],
        output_data: Optional[Dict[str, Any]] = None,
    ):
        """Log span to local storage."""
        trace_id = getattr(trace, 'trace_id', getattr(trace, 'id', str(trace)))
        ms_str = f"{latency_ms:.2f}ms" if latency_ms is not None else "N/A"
        logger.debug(f"Span: {trace_id}/{name} - {ms_str} - {'ERROR: ' + error if error else 'OK'}")

        # Store in DB if session available.
        if self.session_factory:
            try:
                from ..db.repository import TraceLogRepository
                session = self.session_factory()
                try:
                    repo = TraceLogRepository(session)
                    repo.create(
                        trace_id=trace_id,
                        span_name=name,
                        input_data=input_data,
                        output_data=output_data,
                        latency_ms=latency_ms,
                        error=error,
                        commit=True,
                    )
                finally:
                    session.close()
            except Exception as e:
                logger.error(f"Failed to save trace to DB: {e}")
        elif self.db_session:
            try:
                from ..db.repository import TraceLogRepository
                repo = TraceLogRepository(self.db_session)
                in_tx = bool(self.db_session.in_transaction()) if hasattr(self.db_session, "in_transaction") else False
                repo.create(
                    trace_id=trace_id,
                    span_name=name,
                    input_data=input_data,
                    output_data=output_data,
                    latency_ms=latency_ms,
                    error=error,
                    commit=not in_tx,
                )
            except Exception as e:
                logger.error(f"Failed to save trace to DB: {e}")

    def flush(self):
        """Flush pending traces."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")

    # ==================== Prompt Registry ====================

    def get_prompt(self, name: str, label: str = "production"):
        """Fetch a prompt from Langfuse by name and label. Returns None if unavailable."""
        key = (name, label)
        now = time.monotonic()
        if self._missing_prompt_until.get(key, 0.0) > now:
            return None
        cached = self._prompt_cache.get(key)
        if cached and cached[0] > now:
            return cached[1]

        if self.enabled and self.langfuse:
            try:
                prompt = self.langfuse.get_prompt(name, label=label)
                self._prompt_cache[key] = (now + self._prompt_cache_ttl_seconds, prompt)
                self._missing_prompt_until.pop(key, None)
                return prompt
            except Exception as e:
                logger.debug(f"Langfuse prompt '{name}' not found ({label}): {e}")
                self._missing_prompt_until[key] = now + self._missing_prompt_ttl_seconds
        return None

    def list_prompts(self) -> List[dict]:
        """List all prompts from Langfuse."""
        if self.enabled and self.langfuse:
            try:
                api = getattr(self.langfuse, "api", None)
                if api and hasattr(api, "prompts"):
                    result = api.prompts.list()
                    prompts = result.data if hasattr(result, "data") else []
                    return [
                        {"name": p.name, "labels": getattr(p, "labels", []), "versions": getattr(p, "versions", [])}
                        for p in prompts
                    ]
            except Exception as e:
                logger.error(f"Failed to list Langfuse prompts: {e}")
        return []

    def create_prompt(self, name: str, prompt: str, labels: Optional[List[str]] = None) -> Optional[dict]:
        """Register a prompt in Langfuse with optional labels."""
        if self.enabled and self.langfuse:
            try:
                sanitized = [l[:36] for l in (labels or ["draft"])]
                result = self.langfuse.create_prompt(
                    name=name,
                    prompt=prompt,
                    labels=sanitized,
                    type="text",
                )
                return {"name": name, "version": getattr(result, "version", None), "labels": sanitized}
            except Exception as e:
                logger.error(f"Failed to create Langfuse prompt '{name}': {e}")
        return None

    def update_prompt_label(self, name: str, version: int, new_label: str) -> bool:
        """Update a prompt version's label (e.g., set to 'production')."""
        if self.enabled and self.langfuse:
            try:
                sanitized_label = (new_label or "")[:36]
                self.langfuse.update_prompt(name=name, version=version, new_label=sanitized_label)
                return True
            except Exception as e:
                logger.error(f"Failed to update Langfuse prompt label: {e}")
        return False


class LangfuseTraceWrapper:
    """Wraps a Langfuse v3 root span to provide a trace-like interface."""

    def __init__(self, trace_id, name, root_span, langfuse, tags=None, metadata=None):
        self.trace_id = trace_id
        self.id = trace_id
        self.name = name
        self.root_span = root_span
        self.langfuse = langfuse
        self.tags = tags or []
        self.metadata = metadata or {}

    def record_child_span(self, name, input_data=None, latency_ms=None, error=None, output_data=None):
        """Record a child span nested under the root span."""
        try:
            child = self.root_span.start_span(
                name=name,
                input=input_data,
            )
            child.update(
                output=output_data or {"latency_ms": latency_ms},
                level="ERROR" if error else "DEFAULT",
                status_message="error" if error else "success",
            )
            child.end()
        except Exception as e:
            logger.error(f"Failed to record Langfuse child span: {e}")

    def span(self, name: str, input: Optional[Dict] = None):
        """Create a child span on the root span."""
        try:
            return self.root_span.start_span(name=name, input=input)
        except Exception:
            return MockSpan(name, input)

    def score(self, name: str, value: float, comment: Optional[str] = None):
        """Record a score on this trace."""
        try:
            self.root_span.score_trace(
                name=name,
                value=value,
                comment=comment,
                data_type="NUMERIC",
            )
        except Exception as e:
            logger.error(f"Failed to record score: {e}")

    def update(self, **kwargs):
        """Update trace-level fields (input, output, tags)."""
        try:
            self.root_span.update_trace(**kwargs)
        except Exception as e:
            logger.error(f"Failed to update Langfuse trace: {e}")

    def end(self):
        """End the root span."""
        try:
            self.root_span.end()
        except Exception:
            pass


class MockTrace:
    """Mock trace for fallback mode."""

    def __init__(
        self,
        trace_id: str,
        name: str,
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        db_session=None
    ):
        self.id = trace_id
        self.trace_id = trace_id
        self.name = name
        self.tags = tags or []
        self.metadata = metadata or {}
        self.db_session = db_session
        self.spans = []
        self.scores = []

    def record_child_span(self, name, input_data=None, latency_ms=None, error=None, output_data=None):
        """Record a child span (no-op in mock)."""
        self.spans.append({"name": name, "latency_ms": latency_ms, "error": error})

    def span(self, name: str, input: Optional[Dict] = None):
        """Create a mock span."""
        span = MockSpan(name, input)
        self.spans.append(span)
        return span

    def score(self, name: str, value: float, comment: Optional[str] = None):
        """Record a mock score."""
        self.scores.append({"name": name, "value": value, "comment": comment})
        logger.info(f"Score: {name}={value}")


class MockSpan:
    """Mock span for fallback mode."""

    def __init__(self, name: str, input_data: Optional[Dict]):
        self.name = name
        self.input = input_data
        self.output = None
        self.status = None
        self.start_time = time.time()

    def end(self, output: Optional[Dict] = None, status_message: str = "success", **kwargs):
        """End the span."""
        self.output = output
        self.status = status_message
        duration = (time.time() - self.start_time) * 1000
        logger.debug(f"Span ended: {self.name} - {duration:.2f}ms - {status_message}")


def traced(span_name: str):
    """Decorator for tracing functions."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Look for instrumentation in kwargs or first arg
            instrumentation = kwargs.get('instrumentation')
            trace = kwargs.get('trace')

            if instrumentation and trace:
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    instrumentation.record_span(
                        trace,
                        span_name,
                        output_data={"status": "success"},
                        latency_ms=(time.time() - start) * 1000,
                    )
                    return result
                except Exception as e:
                    instrumentation.record_span(
                        trace,
                        span_name,
                        error=str(e),
                        latency_ms=(time.time() - start) * 1000,
                    )
                    raise
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

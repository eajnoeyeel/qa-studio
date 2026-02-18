"""Langfuse instrumentation wrapper with graceful fallback."""
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class LangfuseInstrumentation:
    """Wrapper for Langfuse tracing with fallback to local logging."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        db_session=None
    ):
        self.enabled = bool(public_key and secret_key)
        self.langfuse = None
        self.db_session = db_session

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
        """Create a new trace."""
        if self.enabled and self.langfuse:
            try:
                return self.langfuse.trace(
                    id=trace_id,
                    name=name,
                    tags=tags or [],
                    metadata=metadata or {},
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
        span_obj = None
        error_msg = None

        try:
            if self.enabled and hasattr(trace, 'span'):
                span_obj = trace.span(
                    name=name,
                    input=input_data,
                )
            yield
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000

            if span_obj:
                try:
                    span_obj.end(
                        output={"latency_ms": latency_ms},
                        status_message="error" if error_msg else "success",
                    )
                except Exception:
                    pass

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
        if self.enabled and hasattr(trace, 'span'):
            try:
                span = trace.span(
                    name=name,
                    input=input_data,
                )
                span.end(
                    output=output_data,
                    status_message="error" if error else "success",
                )
            except Exception as e:
                logger.error(f"Failed to record Langfuse span: {e}")

        self._log_span(trace, name, input_data, latency_ms, error)

    def record_score(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """Record a score."""
        if self.enabled and hasattr(trace, 'score'):
            try:
                trace.score(
                    name=name,
                    value=value,
                    comment=comment,
                )
            except Exception as e:
                logger.error(f"Failed to record Langfuse score: {e}")

        # Log to fallback
        logger.info(f"Score recorded: {name}={value} ({comment})")

    def _log_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]],
        latency_ms: Optional[float],
        error: Optional[str]
    ):
        """Log span to local storage."""
        trace_id = getattr(trace, 'id', str(trace))
        ms_str = f"{latency_ms:.2f}ms" if latency_ms is not None else "N/A"
        logger.debug(f"Span: {trace_id}/{name} - {ms_str} - {'ERROR: ' + error if error else 'OK'}")

        # Store in DB if session available
        if self.db_session:
            try:
                from ..db.repository import TraceLogRepository
                repo = TraceLogRepository(self.db_session)
                repo.create(
                    trace_id=trace_id,
                    span_name=name,
                    input_data=input_data,
                    latency_ms=latency_ms,
                    error=error,
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
        self.name = name
        self.tags = tags or []
        self.metadata = metadata or {}
        self.db_session = db_session
        self.spans = []
        self.scores = []

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

    def end(self, output: Optional[Dict] = None, status_message: str = "success"):
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

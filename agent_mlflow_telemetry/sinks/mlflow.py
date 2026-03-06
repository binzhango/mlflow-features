"""MLflow tracing sink abstraction."""

from __future__ import annotations

import logging
import os
from threading import Lock
import time
from typing import Any, Callable

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

from ..runtime.config import TelemetryConfig
from ..domain.schema import SpanRecord, build_span_attributes

LOGGER = logging.getLogger(__name__)

_COMPONENT_TO_SPAN_TYPE = {
    "agent": SpanType.AGENT,
    "chain": SpanType.CHAIN,
    "tool": SpanType.TOOL,
    "llm": SpanType.CHAT_MODEL,
    "retriever": SpanType.RETRIEVER,
}

_STATUS_TO_MLFLOW = {
    "ok": SpanStatusCode.OK,
    "error": SpanStatusCode.ERROR,
    "cancelled": SpanStatusCode.ERROR,
    "timeout": SpanStatusCode.ERROR,
}


class MLflowSink:
    """MLflow tracing sink with bounded retry and fail-open behavior."""

    def __init__(
        self,
        config: TelemetryConfig,
        *,
        mlflow_module: Any | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.1,
    ) -> None:
        self._config = config
        self._mlflow = mlflow_module or mlflow
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._active_spans: dict[str, Any] = {}
        self._lock = Lock()

        if self._config.mlflow_tracking_uri:
            self._mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
        if self._config.mlflow_experiment:
            self._mlflow.set_experiment(self._config.mlflow_experiment)

    @property
    def config(self) -> TelemetryConfig:
        return self._config

    def start_span(self, span: SpanRecord) -> None:
        """Start span in MLflow tracing backend."""

        if not self._config.enabled:
            return

        attrs = build_span_attributes(span)
        parent = None
        if span.parent_span_id:
            with self._lock:
                parent = self._active_spans.get(span.parent_span_id)
        is_root = parent is None
        inputs = self._span_inputs(span=span, attributes=attrs)
        attrs = self._mlflow_attributes(attrs, span)
        metadata = self._trace_metadata(span=span, is_root=is_root)
        if is_root and (run_id := self._active_run_id()):
            metadata = metadata or {}
            metadata[TraceMetadataKey.SOURCE_RUN] = run_id

        def _op() -> Any:
            live_span = self._mlflow.start_span_no_context(
                name=span.name or span.operation,
                span_type=_COMPONENT_TO_SPAN_TYPE.get(span.component, SpanType.UNKNOWN),
                parent_span=parent,
                inputs=inputs,
                attributes={},
                metadata=metadata,
            )
            live_span.set_attributes(attrs)
            return live_span

        live_span = self._retry(_op, action=f"start_span:{span.span_id}")
        if live_span is not None:
            with self._lock:
                self._active_spans[span.span_id] = live_span

    def end_span(self, span: SpanRecord) -> None:
        """End span in MLflow tracing backend."""

        if not self._config.enabled:
            return

        with self._lock:
            live_span = self._active_spans.pop(span.span_id, None)
        if live_span is None:
            LOGGER.warning("end_span called for unknown span_id=%s", span.span_id)
            return

        attrs = build_span_attributes(span)
        status = _STATUS_TO_MLFLOW.get(span.status, SpanStatusCode.UNSET)
        outputs = self._span_outputs(span=span, attributes=attrs)
        attrs = self._mlflow_attributes(attrs, span)

        def _op() -> None:
            live_span.end(outputs=outputs, attributes=attrs, status=status)

        self._retry(_op, action=f"end_span:{span.span_id}")

    def record_event(self, span_id: str, name: str, attributes: dict[str, object] | None = None) -> None:
        """Record event under a span in MLflow tracing backend."""

        if not self._config.enabled:
            return

        with self._lock:
            live_span = self._active_spans.get(span_id)
        if live_span is None:
            LOGGER.warning("record_event called for unknown span_id=%s", span_id)
            return

        event = SpanEvent(name=name, attributes=(attributes or {}))

        def _op() -> None:
            live_span.add_event(event)

        self._retry(_op, action=f"record_event:{span_id}:{name}")

    def _retry(self, op: Callable[[], Any], *, action: str) -> Any | None:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return op()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                is_last = attempt >= self._max_retries
                retryable = self._is_retryable(exc)
                if is_last or not retryable:
                    break
                if self._retry_backoff_seconds > 0:
                    time.sleep(self._retry_backoff_seconds * (attempt + 1))

        assert last_error is not None
        message = f"telemetry action failed: {action} ({type(last_error).__name__}: {last_error})"
        if self._config.fail_open:
            LOGGER.warning(message)
            return None
        raise RuntimeError(message) from last_error

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True
        text = str(exc).lower()
        return any(token in text for token in ("timeout", "tempor", "connection", "unavailable", "503"))

    @staticmethod
    def _span_inputs(*, span: SpanRecord, attributes: dict[str, Any]) -> Any | None:
        if "_mlflow_inputs" in attributes:
            return attributes.pop("_mlflow_inputs")
        if span.prompt_text is not None:
            return span.prompt_text
        return attributes.get("prompt_text")

    @staticmethod
    def _span_outputs(*, span: SpanRecord, attributes: dict[str, Any]) -> Any | None:
        if "_mlflow_outputs" in attributes:
            return attributes.pop("_mlflow_outputs")
        if span.response_text is not None:
            return span.response_text
        return attributes.get("response_text")

    @staticmethod
    def _trace_metadata(*, span: SpanRecord, is_root: bool) -> dict[str, str] | None:
        if not is_root:
            return None

        metadata: dict[str, str] = {}
        if span.session_id:
            metadata[TraceMetadataKey.TRACE_SESSION] = span.session_id
        elif span.trace_id:
            # Keep session column non-empty even when caller does not propagate context.
            metadata[TraceMetadataKey.TRACE_SESSION] = span.trace_id

        if span.user_id:
            metadata[TraceMetadataKey.TRACE_USER] = span.user_id
        elif user := os.getenv("AGENT_TELEMETRY_USER_ID") or os.getenv("USER"):
            metadata[TraceMetadataKey.TRACE_USER] = user
        return metadata or None

    @staticmethod
    def _mlflow_attributes(attributes: dict[str, Any], span: SpanRecord) -> dict[str, Any]:
        payload = {k: v for k, v in attributes.items() if not k.startswith("_mlflow_")}
        if span.model_name:
            payload.setdefault(SpanAttributeKey.MODEL, span.model_name)
        if span.provider:
            payload.setdefault(SpanAttributeKey.MODEL_PROVIDER, span.provider)

        token_usage: dict[str, int] = {}
        if span.input_tokens is not None:
            token_usage["input_tokens"] = span.input_tokens
        if span.output_tokens is not None:
            token_usage["output_tokens"] = span.output_tokens
        if span.total_tokens is not None:
            token_usage["total_tokens"] = span.total_tokens
        if token_usage:
            payload.setdefault(SpanAttributeKey.CHAT_USAGE, token_usage)
        return payload

    def _active_run_id(self) -> str | None:
        active_run_fn = getattr(self._mlflow, "active_run", None)
        if not callable(active_run_fn):
            return None
        try:
            run = active_run_fn()
        except Exception:  # noqa: BLE001
            return None
        info = getattr(run, "info", None)
        run_id = getattr(info, "run_id", None)
        if run_id is None:
            return None
        text = str(run_id).strip()
        return text or None

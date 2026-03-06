"""ChatModel-style adapter scaffold for llmclient compatibility."""

from __future__ import annotations

import hashlib
import time
from typing import Any
from uuid import uuid4

from ...domain.chat_payloads import (
    build_chat_outputs,
    build_chat_request,
    request_preview_text,
    response_preview_text,
    to_text,
)
from ...runtime.context import get_trace_context
from ...sinks.mlflow import MLflowSink
from ...domain.model_config import MODEL_CONFIG_KEYS, extract_model_config
from ...domain.schema import SpanRecord
from ...domain.translation import (
    emit_tool_call_spans_via_sink,
    extract_additional_kwargs_from_response,
    extract_response_metadata_from_response,
    first_non_empty,
    flatten_metadata,
    infer_provider_name,
    normalize_usage,
)


class TelemetryEnabledClient:
    """Wrapper that instruments sync/async/streaming chat model calls."""

    def __init__(self, client: Any, sink: MLflowSink) -> None:
        self._client = client
        self._sink = sink

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        start = self._start_span(
            operation="llm.invoke",
            payload=args[0] if args else kwargs,
            call_kwargs=kwargs,
        )
        try:
            result = self._client.invoke(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._end_error_span(start, exc)
            raise
        self._end_ok_span(start, result, streamed=False)
        return result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        start = self._start_span(
            operation="llm.ainvoke",
            payload=args[0] if args else kwargs,
            call_kwargs=kwargs,
        )
        try:
            result = await self._client.ainvoke(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._end_error_span(start, exc)
            raise
        self._end_ok_span(start, result, streamed=False)
        return result

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        start = self._start_span(
            operation="llm.stream",
            payload=args[0] if args else kwargs,
            call_kwargs=kwargs,
        )
        chunks: list[Any] = []
        try:
            for chunk in self._client.stream(*args, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as exc:  # noqa: BLE001
            self._end_error_span(start, exc)
            raise
        self._end_ok_span(start, chunks, streamed=True)

    def _start_span(self, *, operation: str, payload: Any, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        ctx = get_trace_context()
        metadata = self._metadata_from_kwargs(call_kwargs=call_kwargs)
        model_config = extract_model_config(self._client_model_config_defaults(), metadata, call_kwargs)
        span_id = str(uuid4())
        trace_id = ctx.trace_id or span_id
        inputs_payload = build_chat_request(payload)
        if model_config:
            inputs_payload = {**inputs_payload, **model_config}
        prompt_text = request_preview_text(inputs_payload)
        attrs: dict[str, Any] = {
            "prompt_hash": self._hash(prompt_text),
            "_mlflow_inputs": inputs_payload,
            "mlflow.message.format": "openai",
        }
        if self._sink.config.log_content:
            attrs["prompt_text"] = prompt_text
        if model_config:
            attrs["model_config"] = model_config
            attrs.update(self._flatten_metadata(model_config, prefix="model_config"))

        span = SpanRecord(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=ctx.span_id,
            component="llm",
            operation=operation,
            status="ok",
            session_id=ctx.session_id or self._first_non_empty(metadata, ("session_id", "conversation_id", "thread_id")),
            root_request_id=ctx.root_request_id,
            user_id=self._first_non_empty(metadata, ("user_id", "user", "actor_id")),
            service_name=self._sink.config.service_name,
            service_version=self._sink.config.service_version,
            environment=self._sink.config.environment,
            provider=self._provider_name(),
            model_name=self._model_name(),
            gateway_route=self._gateway_route(),
            attributes=attrs,
        )
        self._sink.start_span(span)
        return {
            "span": span,
            "start_time_ns": time.perf_counter_ns(),
            "model_config": model_config or None,
        }

    def _end_ok_span(self, start: dict[str, Any], result: Any, *, streamed: bool) -> None:
        token_usage = self._extract_token_usage(result)
        response_metadata = self._extract_response_metadata(result)
        outputs_payload = build_chat_outputs(
            result,
            default_role="assistant",
            model_name=self._model_name(),
            token_usage=token_usage,
            response_metadata=response_metadata,
        )
        response_text = response_preview_text(outputs_payload)
        attrs: dict[str, Any] = {
            "response_hash": self._hash(response_text),
            "_mlflow_outputs": outputs_payload,
            "mlflow.message.format": "openai",
            "mlflow.chat.messages": outputs_payload.get("messages", []),
        }
        if response_metadata:
            attrs["response_metadata"] = response_metadata
            attrs.update(self._flatten_metadata(response_metadata, prefix="response_metadata"))
        if start.get("model_config"):
            attrs["model_config"] = start["model_config"]
            attrs.update(self._flatten_metadata(start["model_config"], prefix="model_config"))
        if additional_kwargs := self._extract_additional_kwargs(result):
            attrs["additional_kwargs"] = additional_kwargs
        usage_metadata = getattr(result, "usage_metadata", None)
        if isinstance(usage_metadata, dict) and usage_metadata:
            attrs["usage_metadata"] = usage_metadata
        if self._sink.config.log_content:
            attrs["response_text"] = response_text

        span = SpanRecord(
            trace_id=start["span"].trace_id,
            span_id=start["span"].span_id,
            parent_span_id=start["span"].parent_span_id,
            component="llm",
            operation=start["span"].operation,
            status="ok",
            session_id=start["span"].session_id,
            root_request_id=start["span"].root_request_id,
            service_name=self._sink.config.service_name,
            service_version=self._sink.config.service_version,
            environment=self._sink.config.environment,
            provider=self._provider_name(),
            model_name=self._model_name(),
            gateway_route=self._gateway_route(),
            latency_ms=(time.perf_counter_ns() - start["start_time_ns"]) / 1_000_000,
            input_tokens=token_usage.get("input_tokens") if token_usage else None,
            output_tokens=token_usage.get("output_tokens") if token_usage else None,
            total_tokens=token_usage.get("total_tokens") if token_usage else None,
            streamed=streamed,
            attributes=attrs,
        )
        self._emit_tool_call_spans(parent=start["span"], payload=result)
        self._sink.end_span(span)

    def _end_error_span(self, start: dict[str, Any], error: BaseException) -> None:
        attrs: dict[str, Any] = {}
        if start.get("model_config"):
            attrs["model_config"] = start["model_config"]
            attrs.update(self._flatten_metadata(start["model_config"], prefix="model_config"))
        span = SpanRecord(
            trace_id=start["span"].trace_id,
            span_id=start["span"].span_id,
            parent_span_id=start["span"].parent_span_id,
            component="llm",
            operation=start["span"].operation,
            status="error",
            session_id=start["span"].session_id,
            root_request_id=start["span"].root_request_id,
            service_name=self._sink.config.service_name,
            service_version=self._sink.config.service_version,
            environment=self._sink.config.environment,
            provider=self._provider_name(),
            model_name=self._model_name(),
            gateway_route=self._gateway_route(),
            latency_ms=(time.perf_counter_ns() - start["start_time_ns"]) / 1_000_000,
            error_type=type(error).__name__,
            error_message=str(error),
            attributes=attrs,
        )
        self._sink.end_span(span)

    def _emit_tool_call_spans(self, *, parent: SpanRecord, payload: Any) -> None:
        emit_tool_call_spans_via_sink(
            sink=self._sink,
            parent_trace_id=parent.trace_id,
            parent_span_id=parent.span_id,
            payload=payload,
            session_id=parent.session_id,
            root_request_id=parent.root_request_id,
            user_id=parent.user_id,
            provider=parent.provider,
            model_name=parent.model_name,
            gateway_route=parent.gateway_route,
            endpoint=parent.endpoint,
        )

    def _model_name(self) -> str | None:
        model = getattr(self._client, "model", None)
        if model is None:
            model = getattr(self._client, "model_name", None)
        return str(model) if model is not None else None

    def _provider_name(self) -> str | None:
        inferred = infer_provider_name(class_name=self._client.__class__.__name__)
        if inferred is not None:
            return inferred
        provider = getattr(self._client, "provider", None)
        return str(provider) if provider is not None else None

    def _gateway_route(self) -> str | None:
        route = getattr(self._client, "gateway_route", None)
        return str(route) if route else None

    def _client_model_config_defaults(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for key in MODEL_CONFIG_KEYS:
            value = getattr(self._client, key, None)
            if value is not None:
                defaults[key] = value
        return defaults

    @staticmethod
    def _to_text(payload: Any) -> str:
        return to_text(payload)

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_token_usage(result: Any) -> dict[str, int] | None:
        usage = getattr(result, "usage_metadata", None)
        if usage is None:
            response_meta = getattr(result, "response_metadata", None)
            if isinstance(response_meta, dict):
                usage = response_meta.get("token_usage") or response_meta.get("usage") or response_meta
        if not isinstance(usage, dict):
            return None
        return normalize_usage(usage)

    @staticmethod
    def _extract_response_metadata(result: Any) -> dict[str, Any] | None:
        return extract_response_metadata_from_response(result)

    @staticmethod
    def _extract_additional_kwargs(result: Any) -> dict[str, Any] | None:
        return extract_additional_kwargs_from_response(result)

    @staticmethod
    def _flatten_metadata(metadata: dict[str, Any], *, prefix: str) -> dict[str, Any]:
        return flatten_metadata(metadata, prefix=prefix)

    @staticmethod
    def _metadata_from_kwargs(call_kwargs: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if isinstance(call_kwargs.get("metadata"), dict):
            metadata.update(call_kwargs["metadata"])
        config = call_kwargs.get("config")
        if isinstance(config, dict) and isinstance(config.get("metadata"), dict):
            metadata.update(config["metadata"])
        return metadata

    @staticmethod
    def _first_non_empty(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        return first_non_empty(source, keys)


def create_telemetry_client(client: Any, sink: MLflowSink) -> TelemetryEnabledClient:
    """Create a telemetry-enabled ChatModel-compatible wrapper."""
    return TelemetryEnabledClient(client=client, sink=sink)


InstrumentedLLMClient = TelemetryEnabledClient
wrap_llmclient = create_telemetry_client

"""LangChain callback handler scaffold for telemetry."""

from __future__ import annotations

import hashlib
from typing import Callable
from typing import Any

from mlflow.entities import LiveSpan
from mlflow.entities import SpanType
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.fluent import start_span_no_context as _mlflow_start_span_no_context

from ...domain.chat_payloads import (
    build_chat_outputs,
    build_chat_request,
    extract_tool_calls,
    request_preview_text,
    response_preview_text,
)
from ...sinks.mlflow import MLflowSink
from ...domain.model_config import extract_model_config, extract_model_config_from_serialized_repr
from ...domain.translation import (
    emit_tool_call_spans_via_sink,
    extract_additional_kwargs_from_response,
    extract_response_metadata_from_response,
    extract_token_usage_from_response,
    flatten_metadata,
    infer_model_name,
    infer_provider_name,
)


class LangChainTelemetryTracer(MlflowLangchainTracer):
    """
    MLflow-compatible custom LangChain tracer.

    This intentionally subclasses `MlflowLangchainTracer` so it can be used with
    the same autolog callback injection pattern as official MLflow LangChain tracing.
    """

    def __init__(
        self,
        *,
        sink: MLflowSink | None = None,
        static_attributes: dict[str, Any] | None = None,
        span_processor: Callable[[LiveSpan, dict[str, Any]], None] | None = None,
        log_content: bool | None = None,
        prediction_context: Any | None = None,
        run_inline: bool = False,
    ) -> None:
        super().__init__(prediction_context=prediction_context, run_inline=run_inline)
        # Backward-compat only: legacy bootstrap passes runtime.sink.
        self._sink = sink
        self._static_attributes = dict(static_attributes or {})
        self._span_processor = span_processor
        self._log_content = sink.config.log_content if (log_content is None and sink is not None) else (
            True if log_content is None else bool(log_content)
        )
        self._run_state: dict[str, dict[str, Any]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._remember_llm_run_state(
            run_id=run_id,
            serialized=serialized,
            invocation_params=kwargs.get("invocation_params") or {},
            metadata=metadata or {},
        )
        super().on_llm_start(serialized, prompts, run_id=run_id, metadata=metadata, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._remember_llm_run_state(
            run_id=run_id,
            serialized=serialized,
            invocation_params=kwargs.get("invocation_params") or {},
            metadata=metadata or {},
        )
        super().on_chat_model_start(serialized, messages, run_id=run_id, metadata=metadata, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        try:
            super().on_llm_end(response, run_id=run_id, **kwargs)
        finally:
            self._run_state.pop(str(run_id), None)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        try:
            super().on_llm_error(error, run_id=run_id, **kwargs)
        finally:
            self._run_state.pop(str(run_id), None)

    def _remember_llm_run_state(
        self,
        *,
        run_id: Any,
        serialized: dict[str, Any],
        invocation_params: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        model_config = extract_model_config(
            extract_model_config_from_serialized_repr(serialized.get("repr")),
            metadata,
            invocation_params,
        )
        model_name = infer_model_name(
            serialized=serialized,
            invocation_params=invocation_params,
            metadata=metadata,
        )
        provider = infer_provider_name(serialized=serialized, invocation_params=invocation_params)
        self._run_state[str(run_id)] = {
            "model_config": model_config or None,
            "model_name": model_name,
            "provider": provider,
        }

    def _start_span(
        self,
        span_name: str,
        parent_run_id: Any,
        span_type: str,
        run_id: Any,
        inputs: str | dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> LiveSpan:
        attrs = dict(attributes or {})
        normalized_inputs = inputs
        if span_type in (SpanType.CHAT_MODEL, SpanType.LLM):
            normalized_inputs = build_chat_request(inputs)
            run_state = self._run_state.get(str(run_id), {})
            model_config = run_state.get("model_config")
            if model_config and isinstance(normalized_inputs, dict):
                normalized_inputs = {**normalized_inputs, **model_config}

            prompt_text = request_preview_text(normalized_inputs)
            if prompt_text and self._log_content:
                attrs["prompt_text"] = prompt_text
            if prompt_text:
                attrs["prompt_hash"] = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if model_config:
                attrs["model_config"] = model_config
                attrs.update(flatten_metadata(model_config, prefix="model_config"))
            attrs[SpanAttributeKey.MESSAGE_FORMAT] = "openai"

        span = super()._start_span(
            span_name=span_name,
            parent_run_id=parent_run_id,
            span_type=span_type,
            run_id=run_id,
            inputs=normalized_inputs,
            attributes=attrs,
        )
        run_state = self._run_state.get(str(run_id), {})
        if model_name := run_state.get("model_name"):
            span.set_attribute(SpanAttributeKey.MODEL, model_name)
        if provider := run_state.get("provider"):
            span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, provider)
        for key, value in self._static_attributes.items():
            span.set_attribute(key, value)
        if self._span_processor is not None:
            self._span_processor(
                span,
                {
                    "span_name": span_name,
                    "span_type": span_type,
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
                },
            )
        return span

    def _end_span(
        self,
        run_id: Any,
        span: LiveSpan,
        outputs: Any = None,
        attributes: dict[str, Any] | None = None,
        status: Any = None,
    ) -> None:
        attrs = dict(attributes or {})
        normalized_outputs = outputs

        if span.span_type in (SpanType.CHAT_MODEL, SpanType.LLM) and outputs is not None:
            run_state = self._run_state.get(str(run_id), {})
            model_config = run_state.get("model_config")
            token_usage = extract_token_usage_from_response(outputs)
            response_metadata = extract_response_metadata_from_response(outputs)
            model_name = span.attributes.get(SpanAttributeKey.MODEL)
            model_name_text = str(model_name) if model_name else None
            normalized_outputs = build_chat_outputs(
                outputs,
                default_role="assistant",
                model_name=model_name_text,
                token_usage=token_usage,
                response_metadata=response_metadata,
            )
            response_text = response_preview_text(normalized_outputs)
            if response_text and self._log_content:
                attrs["response_text"] = response_text
            if response_text:
                attrs["response_hash"] = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
            attrs[SpanAttributeKey.MESSAGE_FORMAT] = "openai"
            attrs["mlflow.chat.messages"] = normalized_outputs.get("messages", [])
            if model_config:
                attrs["model_config"] = model_config
                attrs.update(flatten_metadata(model_config, prefix="model_config"))
            if response_metadata:
                attrs["response_metadata"] = response_metadata
                attrs.update(flatten_metadata(response_metadata, prefix="response_metadata"))
            if additional_kwargs := extract_additional_kwargs_from_response(outputs):
                attrs["additional_kwargs"] = additional_kwargs
            if token_usage:
                attrs.setdefault(SpanAttributeKey.CHAT_USAGE, token_usage)
            self._emit_tool_call_spans(parent_span=span, payload=outputs)

        try:
            if status is None:
                super()._end_span(
                    run_id=run_id,
                    span=span,
                    outputs=normalized_outputs,
                    attributes=attrs or None,
                )
                return

            super()._end_span(
                run_id=run_id,
                span=span,
                outputs=normalized_outputs,
                attributes=attrs or None,
                status=status,
            )
        finally:
            self._run_state.pop(str(run_id), None)

    def _emit_tool_call_spans(self, *, parent_span: LiveSpan, payload: Any) -> None:
        if self._sink is not None:
            trace_id = getattr(parent_span, "trace_id", None)
            span_id = getattr(parent_span, "span_id", None)
            if trace_id and span_id:
                emit_tool_call_spans_via_sink(
                    sink=self._sink,
                    parent_trace_id=str(trace_id),
                    parent_span_id=str(span_id),
                    payload=payload,
                    provider=str(parent_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER))
                    if parent_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER)
                    else None,
                    model_name=str(parent_span.attributes.get(SpanAttributeKey.MODEL))
                    if parent_span.attributes.get(SpanAttributeKey.MODEL)
                    else None,
                )
                return

        for call in extract_tool_calls(payload):
            call_name = str(call.get("name") or "unknown")
            call_type = str(call.get("type") or "tool")
            arguments = call.get("arguments")
            operation = f"tool_call.{call_type}.{call_name}"
            call_attrs: dict[str, Any] = {
                "tool_call_type": call_type,
                "tool_call_name": call_name,
                "tool_call": call.get("raw", call),
            }
            if call.get("id"):
                call_attrs["tool_call_id"] = call["id"]
            if arguments is not None:
                call_attrs["tool_call_arguments"] = arguments

            child_span = start_span_no_context(
                name=operation,
                span_type=SpanType.TOOL,
                parent_span=parent_span,
                inputs={
                    "type": call_type,
                    "name": call_name,
                    "arguments": arguments,
                },
                attributes={},
            )
            child_span.end(
                outputs={"tool_call": call.get("raw", call)},
                attributes=call_attrs,
            )


CustomLangchainTracer = LangChainTelemetryTracer


def start_span_no_context(*args: Any, **kwargs: Any) -> Any:
    """Compat bridge so legacy monkeypatch path remains effective in tests/callers."""
    try:
        from ... import langchain_callback as legacy_callback

        maybe_fn = getattr(legacy_callback, "start_span_no_context", None)
        if callable(maybe_fn):
            return maybe_fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        pass
    return _mlflow_start_span_no_context(*args, **kwargs)

"""LangChain callback handler scaffold for telemetry."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from mlflow.entities import LiveSpan
from mlflow.entities import SpanType
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from mlflow.tracing.constant import SpanAttributeKey

from .chat_payloads import (
    build_chat_outputs,
    build_chat_request,
    extract_tool_calls,
    request_preview_text,
    response_preview_text,
    to_text,
)
from .context import get_trace_context
from .mlflow_sink import MLflowSink
from .model_config import extract_model_config, extract_model_config_from_serialized_repr
from .schema import SpanRecord


@dataclass(slots=True)
class _ActiveRun:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    component: str
    operation: str
    start_time_ns: int
    prompt_text: str | None = None
    provider: str | None = None
    model_name: str | None = None
    gateway_route: str | None = None
    endpoint: str | None = None
    model_config: dict[str, Any] | None = None
    session_id: str | None = None
    root_request_id: str | None = None
    user_id: str | None = None


class TelemetryCallbackHandler(BaseCallbackHandler):
    """Callback handler mapping LangChain events to canonical spans."""

    def __init__(self, sink: MLflowSink) -> None:
        self._sink = sink
        self._runs: dict[str, _ActiveRun] = {}

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> Any:
        self._start_run("chain", serialized, kwargs, payload=inputs)
        return None

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> Any:
        self._end_run(kwargs, status="ok", payload=outputs)
        return None

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
        self._end_run(kwargs, status="error", error=error)
        return None

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        self._start_run("tool", serialized, kwargs, payload=input_str)
        return None

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        self._end_run(kwargs, status="ok", payload=output)
        return None

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        self._end_run(kwargs, status="error", error=error)
        return None

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        self._start_run("llm", serialized, kwargs, payload=prompts)
        return None

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> Any:
        self._start_run("llm", serialized, kwargs, payload=messages)
        return None

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        token_usage = self._extract_token_usage(response)
        self._end_run(kwargs, status="ok", payload=response, token_usage=token_usage)
        return None

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        self._end_run(kwargs, status="error", error=error)
        return None

    def on_retriever_start(self, serialized: dict[str, Any], query: str, **kwargs: Any) -> Any:
        self._start_run("retriever", serialized, kwargs, payload=query)
        return None

    def on_retriever_end(self, documents: list[Any], **kwargs: Any) -> Any:
        self._end_run(kwargs, status="ok", payload=documents)
        return None

    def on_retriever_error(self, error: BaseException, **kwargs: Any) -> Any:
        self._end_run(kwargs, status="error", error=error)
        return None

    def _start_run(
        self,
        component: str,
        serialized: dict[str, Any],
        kwargs: dict[str, Any],
        *,
        payload: Any,
    ) -> None:
        run_id = str(kwargs.get("run_id"))
        if run_id == "None":
            return
        parent_run_id = kwargs.get("parent_run_id")
        parent_key = str(parent_run_id) if parent_run_id is not None else None

        ctx = get_trace_context()
        metadata = self._metadata(kwargs)
        parent_active = self._runs.get(parent_key) if parent_key else None
        trace_id = (
            parent_active.trace_id
            if parent_active is not None
            else (ctx.trace_id if ctx.trace_id else run_id)
        )
        parent_span_id = (
            parent_active.span_id
            if parent_active is not None
            else (ctx.span_id if parent_key is None else parent_key)
        )
        operation = self._operation_name(component=component, serialized=serialized)
        input_payload = build_chat_request(payload) if component == "llm" else payload
        prompt_text = request_preview_text(input_payload) if component == "llm" else self._to_text(payload)

        invocation_params = kwargs.get("invocation_params") or {}
        provider = self._provider_name(serialized=serialized, invocation_params=invocation_params)
        model_name = self._model_name(serialized=serialized, invocation_params=invocation_params, metadata=metadata)
        gateway_route = (
            invocation_params.get("gateway_route")
            or self._first_non_empty(metadata, ("gateway_route", "route", "path"))
        )
        endpoint = self._endpoint(invocation_params=invocation_params, metadata=metadata)
        model_config = extract_model_config(
            extract_model_config_from_serialized_repr(serialized.get("repr")),
            metadata,
            invocation_params,
        )
        if component == "llm" and model_config and isinstance(input_payload, dict):
            input_payload = {**input_payload, **model_config}
            prompt_text = request_preview_text(input_payload)
        session_id = ctx.session_id or self._first_non_empty(
            metadata,
            ("session_id", "conversation_id", "thread_id", "chat_id"),
        )
        root_request_id = ctx.root_request_id or self._first_non_empty(
            metadata,
            ("root_request_id", "request_id"),
        )
        user_id = self._user_id(kwargs=kwargs, metadata=metadata)

        start = _ActiveRun(
            trace_id=trace_id,
            span_id=run_id,
            parent_span_id=parent_span_id,
            component=component,
            operation=operation,
            start_time_ns=time.perf_counter_ns(),
            prompt_text=prompt_text,
            provider=provider,
            model_name=model_name,
            gateway_route=gateway_route,
            endpoint=endpoint,
            model_config=model_config or None,
            session_id=session_id,
            root_request_id=root_request_id,
            user_id=user_id,
        )
        self._runs[run_id] = start

        attrs: dict[str, Any] = {}
        if self._sink.config.log_content and prompt_text:
            attrs["prompt_text"] = prompt_text
        if prompt_text:
            attrs["prompt_hash"] = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if component == "llm":
            attrs["_mlflow_inputs"] = input_payload
            attrs["mlflow.message.format"] = "openai"
            if model_config:
                attrs["model_config"] = model_config
                attrs.update(self._flatten_metadata(model_config, prefix="model_config"))

        span = SpanRecord(
            trace_id=start.trace_id,
            span_id=start.span_id,
            parent_span_id=start.parent_span_id,
            component=component,
            operation=operation,
            status="ok",
            session_id=start.session_id,
            root_request_id=start.root_request_id,
            user_id=start.user_id,
            service_name=self._sink.config.service_name,
            service_version=self._sink.config.service_version,
            environment=self._sink.config.environment,
            provider=provider,
            model_name=model_name,
            gateway_route=gateway_route,
            endpoint=endpoint,
            attributes=attrs,
        )
        self._sink.start_span(span)

    def _end_run(
        self,
        kwargs: dict[str, Any],
        *,
        status: str,
        payload: Any | None = None,
        error: BaseException | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        run_id = str(kwargs.get("run_id"))
        start = self._runs.pop(run_id, None)
        if start is None:
            return

        elapsed_ms = (time.perf_counter_ns() - start.start_time_ns) / 1_000_000
        output_payload = None
        response_metadata: dict[str, Any] | None = None
        if payload is not None:
            if start.component == "llm":
                response_metadata = self._extract_response_metadata(payload)
                output_payload = build_chat_outputs(
                    payload,
                    default_role="assistant",
                    model_name=start.model_name,
                    token_usage=token_usage,
                    response_metadata=response_metadata,
                )
                response_text = response_preview_text(output_payload)
            elif start.component == "chain":
                output_payload = build_chat_outputs(
                    payload,
                    default_role="assistant",
                    model_name=start.model_name,
                )
                response_text = response_preview_text(output_payload)
            else:
                response_text = self._to_text(payload)
        else:
            response_text = None

        attributes: dict[str, Any] = {}
        if status == "error" and error is not None:
            attributes["error_type"] = type(error).__name__
            attributes["error_message"] = str(error)
        if response_text and self._sink.config.log_content:
            attributes["response_text"] = response_text
        if response_text:
            attributes["response_hash"] = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        if output_payload is not None:
            attributes["_mlflow_outputs"] = output_payload
            attributes["mlflow.message.format"] = "openai"
            attributes["mlflow.chat.messages"] = output_payload.get("messages", [])
        if start.component == "llm":
            if start.model_config:
                attributes["model_config"] = start.model_config
                attributes.update(self._flatten_metadata(start.model_config, prefix="model_config"))
            if response_metadata:
                attributes["response_metadata"] = response_metadata
                attributes.update(self._flatten_metadata(response_metadata, prefix="response_metadata"))
            if additional_kwargs := self._extract_additional_kwargs(payload):
                attributes["additional_kwargs"] = additional_kwargs
            if payload is not None:
                self._emit_tool_call_spans(start=start, payload=payload)

        input_tokens = token_usage.get("input_tokens") if token_usage else None
        output_tokens = token_usage.get("output_tokens") if token_usage else None
        total_tokens = token_usage.get("total_tokens") if token_usage else None

        span = SpanRecord(
            trace_id=start.trace_id,
            span_id=start.span_id,
            parent_span_id=start.parent_span_id,
            component=start.component,
            operation=start.operation,
            status=status,
            session_id=start.session_id,
            root_request_id=start.root_request_id,
            user_id=start.user_id,
            service_name=self._sink.config.service_name,
            service_version=self._sink.config.service_version,
            environment=self._sink.config.environment,
            provider=start.provider,
            model_name=start.model_name,
            gateway_route=start.gateway_route,
            endpoint=start.endpoint,
            latency_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
            attributes=attributes,
        )
        self._sink.end_span(span)

    def _emit_tool_call_spans(self, *, start: _ActiveRun, payload: Any) -> None:
        for idx, call in enumerate(extract_tool_calls(payload)):
            call_name = call.get("name") or "unknown"
            call_type = call.get("type") or "tool"
            arguments = call.get("arguments")
            operation = f"tool_call.{call_type}.{call_name}"
            span_id = f"{start.span_id}:toolcall:{idx}"
            call_attrs: dict[str, Any] = {
                "tool_call_type": call_type,
                "tool_call_name": call_name,
                "tool_call": call.get("raw", call),
            }
            if call.get("id"):
                call_attrs["tool_call_id"] = call["id"]
            if arguments is not None:
                call_attrs["tool_call_arguments"] = arguments

            start_span = SpanRecord(
                trace_id=start.trace_id,
                span_id=span_id,
                parent_span_id=start.span_id,
                component="tool",
                operation=operation,
                status="ok",
                session_id=start.session_id,
                root_request_id=start.root_request_id,
                user_id=start.user_id,
                service_name=self._sink.config.service_name,
                service_version=self._sink.config.service_version,
                environment=self._sink.config.environment,
                provider=start.provider,
                model_name=start.model_name,
                gateway_route=start.gateway_route,
                endpoint=start.endpoint,
                attributes={
                    "_mlflow_inputs": {
                        "type": call_type,
                        "name": call_name,
                        "arguments": arguments,
                    }
                },
            )
            self._sink.start_span(start_span)

            end_attrs = dict(call_attrs)
            end_attrs["_mlflow_outputs"] = {"tool_call": call.get("raw", call)}
            end_span = SpanRecord(
                trace_id=start.trace_id,
                span_id=span_id,
                parent_span_id=start.span_id,
                component="tool",
                operation=operation,
                status="ok",
                session_id=start.session_id,
                root_request_id=start.root_request_id,
                user_id=start.user_id,
                service_name=self._sink.config.service_name,
                service_version=self._sink.config.service_version,
                environment=self._sink.config.environment,
                provider=start.provider,
                model_name=start.model_name,
                gateway_route=start.gateway_route,
                endpoint=start.endpoint,
                latency_ms=0.0,
                attributes=end_attrs,
            )
            self._sink.end_span(end_span)

    @staticmethod
    def _operation_name(component: str, serialized: dict[str, Any]) -> str:
        raw = serialized.get("name") or serialized.get("id") or component
        if isinstance(raw, list):
            raw = ".".join(str(v) for v in raw)
        return f"{component}.{str(raw)}"

    @staticmethod
    def _provider_name(serialized: dict[str, Any], invocation_params: dict[str, Any]) -> str | None:
        if provider := invocation_params.get("provider"):
            return str(provider)
        source = str(serialized.get("id") or serialized.get("name") or "")
        lower = source.lower()
        if "ollama" in lower:
            return "ollama"
        if "openai" in lower:
            return "openai"
        if "anthropic" in lower:
            return "anthropic"
        return None

    @classmethod
    def _model_name(
        cls,
        *,
        serialized: dict[str, Any],
        invocation_params: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str | None:
        model_name = (
            invocation_params.get("model")
            or invocation_params.get("model_name")
            or invocation_params.get("model_id")
            or cls._first_non_empty(metadata, ("model", "model_name", "model_id"))
        )
        if model_name:
            return str(model_name)

        # ChatOllama and similar models often expose model info in the serialized repr.
        serialized_repr = serialized.get("repr")
        if not isinstance(serialized_repr, str):
            return None
        match = re.search(r"model='([^']+)'", serialized_repr)
        if not match:
            return None
        return match.group(1)

    @classmethod
    def _endpoint(cls, *, invocation_params: dict[str, Any], metadata: dict[str, Any]) -> str | None:
        endpoint = (
            invocation_params.get("endpoint")
            or invocation_params.get("base_url")
            or invocation_params.get("api_base")
            or invocation_params.get("url")
            or cls._first_non_empty(metadata, ("endpoint", "base_url", "api_base", "url"))
        )
        return str(endpoint) if endpoint else None

    @classmethod
    def _user_id(cls, *, kwargs: dict[str, Any], metadata: dict[str, Any]) -> str | None:
        user_id = kwargs.get("user_id") or cls._first_non_empty(metadata, ("user_id", "user", "actor_id"))
        return str(user_id) if user_id is not None else None

    @staticmethod
    def _metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return {}

    @staticmethod
    def _first_non_empty(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = source.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _to_text(payload: Any) -> str:
        return to_text(payload)

    @staticmethod
    def _extract_token_usage(response: Any) -> dict[str, int] | None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            return TelemetryCallbackHandler._normalize_usage(usage)
        if isinstance(llm_output, dict):
            if inferred := TelemetryCallbackHandler._usage_from_response_metadata(llm_output):
                return inferred

        # Fallback for integrations that put usage on generated messages instead of llm_output.
        for item in TelemetryCallbackHandler._iter_generation_items(response):
            message = getattr(item, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict):
                return TelemetryCallbackHandler._normalize_usage(usage)
            response_meta = getattr(message, "response_metadata", None)
            if isinstance(response_meta, dict):
                usage = response_meta.get("token_usage") or response_meta.get("usage")
                if isinstance(usage, dict):
                    return TelemetryCallbackHandler._normalize_usage(usage)
                if inferred := TelemetryCallbackHandler._usage_from_response_metadata(response_meta):
                    return inferred
        return None

    @staticmethod
    def _iter_generation_items(response: Any) -> list[Any]:
        generations = getattr(response, "generations", None) or []
        items: list[Any] = []
        for group in generations:
            items.extend(group or [])
        return items

    @staticmethod
    def _extract_response_metadata(response: Any) -> dict[str, Any] | None:
        for item in TelemetryCallbackHandler._iter_generation_items(response):
            message = getattr(item, "message", None)
            response_meta = getattr(message, "response_metadata", None)
            if isinstance(response_meta, dict) and response_meta:
                return response_meta
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict) and llm_output:
            return llm_output
        return None

    @staticmethod
    def _extract_additional_kwargs(response: Any) -> dict[str, Any] | None:
        for item in TelemetryCallbackHandler._iter_generation_items(response):
            message = getattr(item, "message", None)
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict) and additional_kwargs:
                return additional_kwargs
        return None

    @staticmethod
    def _normalize_usage(usage: dict[str, Any]) -> dict[str, int] | None:
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_eval_count")
            or usage.get("input_token_count")
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("eval_count")
            or usage.get("output_token_count")
        )
        total_tokens = usage.get("total_tokens") or usage.get("token_count")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        cleaned: dict[str, int] = {}
        if input_tokens is not None:
            cleaned["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            cleaned["output_tokens"] = int(output_tokens)
        if total_tokens is not None:
            cleaned["total_tokens"] = int(total_tokens)
        return cleaned or None

    @staticmethod
    def _usage_from_response_metadata(response_meta: dict[str, Any]) -> dict[str, int] | None:
        usage = response_meta.get("token_usage") or response_meta.get("usage")
        if isinstance(usage, dict):
            return TelemetryCallbackHandler._normalize_usage(usage)
        return TelemetryCallbackHandler._normalize_usage(response_meta)

    @staticmethod
    def _flatten_metadata(metadata: dict[str, Any], *, prefix: str) -> dict[str, Any]:
        flat: dict[str, Any] = {}

        def _walk(value: Any, path: str) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    if key is None:
                        continue
                    key_text = str(key).strip()
                    if not key_text:
                        continue
                    _walk(nested, f"{path}.{key_text}")
                return
            if isinstance(value, list):
                # Keep list values compact but visible in the attribute table.
                flat[path] = json.dumps(value, default=str, ensure_ascii=True)
                return
            flat[path] = value

        _walk(metadata, prefix)
        return flat


class CustomLangchainTracer(MlflowLangchainTracer):
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
        prediction_context: Any | None = None,
        run_inline: bool = False,
    ) -> None:
        super().__init__(prediction_context=prediction_context, run_inline=run_inline)
        # Backward-compat only: legacy bootstrap passes runtime.sink.
        self._sink = sink
        self._static_attributes = dict(static_attributes or {})
        self._span_processor = span_processor

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
            attrs[SpanAttributeKey.MESSAGE_FORMAT] = "openai"

        span = super()._start_span(
            span_name=span_name,
            parent_run_id=parent_run_id,
            span_type=span_type,
            run_id=run_id,
            inputs=normalized_inputs,
            attributes=attrs,
        )
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
            token_usage = TelemetryCallbackHandler._extract_token_usage(outputs)
            response_metadata = TelemetryCallbackHandler._extract_response_metadata(outputs)
            model_name = span.attributes.get(SpanAttributeKey.MODEL)
            model_name_text = str(model_name) if model_name else None
            normalized_outputs = build_chat_outputs(
                outputs,
                default_role="assistant",
                model_name=model_name_text,
                token_usage=token_usage,
                response_metadata=response_metadata,
            )
            attrs[SpanAttributeKey.MESSAGE_FORMAT] = "openai"
            attrs["mlflow.chat.messages"] = normalized_outputs.get("messages", [])

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

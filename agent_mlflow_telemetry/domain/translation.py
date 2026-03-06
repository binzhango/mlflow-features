"""Shared translation helpers used by telemetry integrations."""

from __future__ import annotations

import json
import re
from typing import Any

from .chat_payloads import extract_tool_calls
from ..sinks.mlflow import MLflowSink
from .schema import SpanRecord


def first_non_empty(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty string-like value from a dictionary."""
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def flatten_metadata(metadata: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Flatten nested dictionaries under a stable dotted prefix."""
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
            flat[path] = json.dumps(value, default=str, ensure_ascii=True)
            return
        flat[path] = value

    _walk(metadata, prefix)
    return flat


def normalize_usage(usage: dict[str, Any]) -> dict[str, int] | None:
    """Normalize provider-specific token counters to input/output/total keys."""
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


def usage_from_response_metadata(response_meta: dict[str, Any]) -> dict[str, int] | None:
    """Extract normalized usage from response metadata dictionaries."""
    usage = response_meta.get("token_usage") or response_meta.get("usage")
    if isinstance(usage, dict):
        return normalize_usage(usage)
    return normalize_usage(response_meta)


def iter_generation_items(response: Any) -> list[Any]:
    """Flatten nested generation groups from LangChain-style LLM responses."""
    generations = getattr(response, "generations", None) or []
    items: list[Any] = []
    for group in generations:
        items.extend(group or [])
    return items


def extract_token_usage_from_response(response: Any) -> dict[str, int] | None:
    """Extract normalized token usage from LangChain-style response objects."""
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        return normalize_usage(usage)
    if isinstance(llm_output, dict):
        if inferred := usage_from_response_metadata(llm_output):
            return inferred

    for item in iter_generation_items(response):
        message = getattr(item, "message", None)
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            return normalize_usage(usage)
        response_meta = getattr(message, "response_metadata", None)
        if isinstance(response_meta, dict):
            usage = response_meta.get("token_usage") or response_meta.get("usage")
            if isinstance(usage, dict):
                return normalize_usage(usage)
            if inferred := usage_from_response_metadata(response_meta):
                return inferred
    return None


def extract_response_metadata_from_response(response: Any) -> dict[str, Any] | None:
    """Extract first non-empty response metadata dictionary."""
    direct = getattr(response, "response_metadata", None)
    if isinstance(direct, dict) and direct:
        return direct

    for item in iter_generation_items(response):
        message = getattr(item, "message", None)
        response_meta = getattr(message, "response_metadata", None)
        if isinstance(response_meta, dict) and response_meta:
            return response_meta
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict) and llm_output:
        return llm_output
    return None


def extract_additional_kwargs_from_response(response: Any) -> dict[str, Any] | None:
    """Extract first non-empty additional_kwargs dictionary."""
    direct = getattr(response, "additional_kwargs", None)
    if isinstance(direct, dict) and direct:
        return direct

    for item in iter_generation_items(response):
        message = getattr(item, "message", None)
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict) and additional_kwargs:
            return additional_kwargs
    return None


def infer_provider_name(
    *,
    invocation_params: dict[str, Any] | None = None,
    serialized: dict[str, Any] | None = None,
    class_name: str | None = None,
) -> str | None:
    """Infer provider from explicit params first, then common class/id markers."""
    invocation_params = invocation_params or {}
    serialized = serialized or {}
    if provider := invocation_params.get("provider"):
        return str(provider)
    source = " ".join(
        part for part in (class_name, str(serialized.get("id") or ""), str(serialized.get("name") or "")) if part
    ).lower()
    if "ollama" in source:
        return "ollama"
    if "openai" in source:
        return "openai"
    if "anthropic" in source:
        return "anthropic"
    return None


def infer_model_name(
    *,
    serialized: dict[str, Any],
    invocation_params: dict[str, Any],
    metadata: dict[str, Any],
) -> str | None:
    """Infer model name from invocation params/metadata, then repr fallback."""
    model_name = (
        invocation_params.get("model")
        or invocation_params.get("model_name")
        or invocation_params.get("model_id")
        or first_non_empty(metadata, ("model", "model_name", "model_id"))
    )
    if model_name:
        return str(model_name)

    serialized_repr = serialized.get("repr")
    if not isinstance(serialized_repr, str):
        return None
    match = re.search(r"model='([^']+)'", serialized_repr)
    if not match:
        return None
    return match.group(1)


def emit_tool_call_spans_via_sink(
    *,
    sink: MLflowSink,
    parent_trace_id: str,
    parent_span_id: str,
    payload: Any,
    session_id: str | None = None,
    root_request_id: str | None = None,
    user_id: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    gateway_route: str | None = None,
    endpoint: str | None = None,
) -> None:
    """Emit normalized tool-call child spans through the sink abstraction."""
    for idx, call in enumerate(extract_tool_calls(payload)):
        call_name = str(call.get("name") or "unknown")
        call_type = str(call.get("type") or "tool")
        arguments = call.get("arguments")
        operation = f"tool_call.{call_type}.{call_name}"
        span_id = f"{parent_span_id}:toolcall:{idx}"
        call_attrs: dict[str, Any] = {
            "tool_call_type": call_type,
            "tool_call_name": call_name,
            "tool_call": call.get("raw", call),
        }
        if call.get("id"):
            call_attrs["tool_call_id"] = call["id"]
        if arguments is not None:
            call_attrs["tool_call_arguments"] = arguments

        sink.start_span(
            SpanRecord(
                trace_id=parent_trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                component="tool",
                operation=operation,
                status="ok",
                session_id=session_id,
                root_request_id=root_request_id,
                user_id=user_id,
                service_name=sink.config.service_name,
                service_version=sink.config.service_version,
                environment=sink.config.environment,
                provider=provider,
                model_name=model_name,
                gateway_route=gateway_route,
                endpoint=endpoint,
                attributes={
                    "_mlflow_inputs": {
                        "type": call_type,
                        "name": call_name,
                        "arguments": arguments,
                    }
                },
            )
        )

        end_attrs = dict(call_attrs)
        end_attrs["_mlflow_outputs"] = {"tool_call": call.get("raw", call)}
        sink.end_span(
            SpanRecord(
                trace_id=parent_trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                component="tool",
                operation=operation,
                status="ok",
                session_id=session_id,
                root_request_id=root_request_id,
                user_id=user_id,
                service_name=sink.config.service_name,
                service_version=sink.config.service_version,
                environment=sink.config.environment,
                provider=provider,
                model_name=model_name,
                gateway_route=gateway_route,
                endpoint=endpoint,
                latency_ms=0.0,
                attributes=end_attrs,
            )
        )

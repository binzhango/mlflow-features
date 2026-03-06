"""Canonical domain models and normalization helpers."""

from .chat_payloads import (
    build_chat_messages,
    build_chat_outputs,
    build_chat_request,
    build_chat_response,
    extract_tool_calls,
    request_preview_text,
    response_preview_text,
    to_text,
)
from .model_config import MODEL_CONFIG_KEYS, extract_model_config, extract_model_config_from_serialized_repr
from .schema import SpanRecord, build_span_attributes
from .translation import (
    emit_tool_call_spans_via_sink,
    extract_additional_kwargs_from_response,
    extract_response_metadata_from_response,
    extract_token_usage_from_response,
    first_non_empty,
    flatten_metadata,
    infer_model_name,
    infer_provider_name,
    normalize_usage,
)

__all__ = [
    "SpanRecord",
    "build_span_attributes",
    "MODEL_CONFIG_KEYS",
    "extract_model_config",
    "extract_model_config_from_serialized_repr",
    "to_text",
    "build_chat_request",
    "build_chat_response",
    "build_chat_messages",
    "build_chat_outputs",
    "extract_tool_calls",
    "request_preview_text",
    "response_preview_text",
    "first_non_empty",
    "flatten_metadata",
    "normalize_usage",
    "extract_token_usage_from_response",
    "extract_response_metadata_from_response",
    "extract_additional_kwargs_from_response",
    "infer_provider_name",
    "infer_model_name",
    "emit_tool_call_spans_via_sink",
]

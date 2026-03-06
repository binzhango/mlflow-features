"""Backward-compatible re-export for shared translation helpers."""

from .domain.translation import (
    emit_tool_call_spans_via_sink,
    extract_additional_kwargs_from_response,
    extract_response_metadata_from_response,
    extract_token_usage_from_response,
    first_non_empty,
    flatten_metadata,
    infer_model_name,
    infer_provider_name,
    iter_generation_items,
    normalize_usage,
    usage_from_response_metadata,
)

__all__ = [
    "first_non_empty",
    "flatten_metadata",
    "normalize_usage",
    "usage_from_response_metadata",
    "iter_generation_items",
    "extract_token_usage_from_response",
    "extract_response_metadata_from_response",
    "extract_additional_kwargs_from_response",
    "infer_provider_name",
    "infer_model_name",
    "emit_tool_call_spans_via_sink",
]

"""Backward-compatible re-export for canonical span schema."""

from .domain.schema import (
    REQUIRED_ATTRIBUTE_KEYS,
    VALID_COMPONENTS,
    VALID_STATUSES,
    SpanRecord,
    build_span_attributes,
)

__all__ = [
    "VALID_COMPONENTS",
    "VALID_STATUSES",
    "REQUIRED_ATTRIBUTE_KEYS",
    "SpanRecord",
    "build_span_attributes",
]

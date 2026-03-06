"""Canonical telemetry schema primitives."""

from dataclasses import dataclass, field
from typing import Any

VALID_COMPONENTS = frozenset({"agent", "chain", "tool", "llm", "retriever"})
VALID_STATUSES = frozenset({"ok", "error", "cancelled", "timeout"})

REQUIRED_ATTRIBUTE_KEYS = (
    "trace_id",
    "span_id",
    "component",
    "operation",
    "status",
)


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


@dataclass(slots=True)
class SpanRecord:
    """Normalized canonical span payload used by sink implementations."""

    trace_id: str
    span_id: str
    component: str
    operation: str
    status: str

    parent_span_id: str | None = None
    name: str | None = None
    session_id: str | None = None
    root_request_id: str | None = None
    user_id: str | None = None

    service_name: str | None = None
    service_version: str | None = None
    environment: str | None = None

    provider: str | None = None
    model_name: str | None = None
    gateway_route: str | None = None
    endpoint: str | None = None

    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    streamed: bool | None = None

    prompt_text: str | None = None
    response_text: str | None = None
    prompt_hash: str | None = None

    error_type: str | None = None
    error_message: str | None = None
    http_status: int | None = None

    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trace_id = _require_non_empty(self.trace_id, field_name="trace_id")
        self.span_id = _require_non_empty(self.span_id, field_name="span_id")
        self.component = _require_non_empty(self.component, field_name="component").lower()
        self.operation = _require_non_empty(self.operation, field_name="operation")
        self.status = _require_non_empty(self.status, field_name="status").lower()

        if self.parent_span_id is not None:
            self.parent_span_id = _require_non_empty(self.parent_span_id, field_name="parent_span_id")

        if self.name is None:
            self.name = self.operation
        else:
            self.name = _require_non_empty(self.name, field_name="name")

        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"component must be one of {sorted(VALID_COMPONENTS)}, got {self.component!r}"
            )
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {self.status!r}")

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")
        if self.total_tokens is not None and self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")

        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens must equal input_tokens + output_tokens when all are provided")

        if self.status == "error" and not (self.error_type or self.error_message):
            raise ValueError("error status requires error_type or error_message")

    def to_attributes(self) -> dict[str, Any]:
        """Build canonical attributes with strict required/optional handling."""

        payload: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "component": self.component,
            "operation": self.operation,
            "status": self.status,
        }

        optional_values = {
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "session_id": self.session_id,
            "root_request_id": self.root_request_id,
            "user_id": self.user_id,
            "service.name": self.service_name,
            "service.version": self.service_version,
            "env": self.environment,
            "provider": self.provider,
            "model_name": self.model_name,
            "gateway_route": self.gateway_route,
            "endpoint": self.endpoint,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "streamed": self.streamed,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "prompt_hash": self.prompt_hash,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "http_status": self.http_status,
        }
        for key, value in optional_values.items():
            if value is not None:
                payload[key] = value

        reserved = set(REQUIRED_ATTRIBUTE_KEYS)
        conflicts = reserved.intersection(self.attributes)
        if conflicts:
            conflict_keys = ", ".join(sorted(conflicts))
            raise ValueError(f"attributes must not override required keys: {conflict_keys}")
        payload.update(self.attributes)
        return payload


def build_span_attributes(span: SpanRecord) -> dict[str, Any]:
    """Helper that builds a canonical attributes dictionary for sink writers."""

    return span.to_attributes()

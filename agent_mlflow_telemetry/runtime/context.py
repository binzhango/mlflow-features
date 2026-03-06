"""Context helpers for trace and span propagation."""

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator

TRACE_ID: ContextVar[str | None] = ContextVar("trace_id", default=None)
SPAN_ID: ContextVar[str | None] = ContextVar("span_id", default=None)
PARENT_SPAN_ID: ContextVar[str | None] = ContextVar("parent_span_id", default=None)
SESSION_ID: ContextVar[str | None] = ContextVar("session_id", default=None)
ROOT_REQUEST_ID: ContextVar[str | None] = ContextVar("root_request_id", default=None)


@dataclass(slots=True)
class TraceContext:
    """Snapshot of current request trace context."""

    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    session_id: str | None = None
    root_request_id: str | None = None


class _Unset:
    pass


UNSET = _Unset()


def get_trace_context() -> TraceContext:
    """Return the active contextvars trace snapshot."""

    return TraceContext(
        trace_id=TRACE_ID.get(),
        span_id=SPAN_ID.get(),
        parent_span_id=PARENT_SPAN_ID.get(),
        session_id=SESSION_ID.get(),
        root_request_id=ROOT_REQUEST_ID.get(),
    )


@contextmanager
def with_trace_context(
    *,
    trace_id: str | None | _Unset = UNSET,
    span_id: str | None | _Unset = UNSET,
    parent_span_id: str | None | _Unset = UNSET,
    session_id: str | None | _Unset = UNSET,
    root_request_id: str | None | _Unset = UNSET,
) -> Iterator[None]:
    """Temporarily set trace context values for nested operations.

    If a field is not provided, its current value is preserved.
    """

    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    if not isinstance(trace_id, _Unset):
        tokens.append((TRACE_ID, TRACE_ID.set(trace_id)))
    if not isinstance(span_id, _Unset):
        tokens.append((SPAN_ID, SPAN_ID.set(span_id)))
    if not isinstance(parent_span_id, _Unset):
        tokens.append((PARENT_SPAN_ID, PARENT_SPAN_ID.set(parent_span_id)))
    if not isinstance(session_id, _Unset):
        tokens.append((SESSION_ID, SESSION_ID.set(session_id)))
    if not isinstance(root_request_id, _Unset):
        tokens.append((ROOT_REQUEST_ID, ROOT_REQUEST_ID.set(root_request_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)

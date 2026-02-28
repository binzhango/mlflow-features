from agent_mlflow_telemetry.context import (
    PARENT_SPAN_ID,
    ROOT_REQUEST_ID,
    SESSION_ID,
    SPAN_ID,
    TRACE_ID,
    TraceContext,
    get_trace_context,
    with_trace_context,
)


def test_with_trace_context_sets_values_within_scope() -> None:
    assert TRACE_ID.get() is None
    assert SPAN_ID.get() is None
    assert PARENT_SPAN_ID.get() is None
    assert SESSION_ID.get() is None
    assert ROOT_REQUEST_ID.get() is None

    with with_trace_context(
        trace_id="t1",
        span_id="s1",
        parent_span_id="p1",
        session_id="sess1",
        root_request_id="r1",
    ):
        assert TRACE_ID.get() == "t1"
        assert SPAN_ID.get() == "s1"
        assert PARENT_SPAN_ID.get() == "p1"
        assert SESSION_ID.get() == "sess1"
        assert ROOT_REQUEST_ID.get() == "r1"

    assert TRACE_ID.get() is None
    assert SPAN_ID.get() is None
    assert PARENT_SPAN_ID.get() is None
    assert SESSION_ID.get() is None
    assert ROOT_REQUEST_ID.get() is None


def test_with_trace_context_restores_outer_values_when_nested() -> None:
    with with_trace_context(trace_id="outer", span_id="outer-span", session_id="outer-session"):
        assert TRACE_ID.get() == "outer"
        assert SPAN_ID.get() == "outer-span"
        assert SESSION_ID.get() == "outer-session"

        with with_trace_context(trace_id="inner", span_id="inner-span", session_id="inner-session"):
            assert TRACE_ID.get() == "inner"
            assert SPAN_ID.get() == "inner-span"
            assert SESSION_ID.get() == "inner-session"

        assert TRACE_ID.get() == "outer"
        assert SPAN_ID.get() == "outer-span"
        assert SESSION_ID.get() == "outer-session"


def test_with_trace_context_preserves_unspecified_fields() -> None:
    with with_trace_context(
        trace_id="outer-trace",
        span_id="outer-span",
        parent_span_id="outer-parent",
        session_id="outer-session",
        root_request_id="outer-root",
    ):
        with with_trace_context(span_id="inner-span"):
            assert TRACE_ID.get() == "outer-trace"
            assert SPAN_ID.get() == "inner-span"
            assert PARENT_SPAN_ID.get() == "outer-parent"
            assert SESSION_ID.get() == "outer-session"
            assert ROOT_REQUEST_ID.get() == "outer-root"


def test_with_trace_context_allows_explicit_none_overrides() -> None:
    with with_trace_context(trace_id="t1", parent_span_id="p1"):
        with with_trace_context(parent_span_id=None):
            assert TRACE_ID.get() == "t1"
            assert PARENT_SPAN_ID.get() is None


def test_get_trace_context_returns_snapshot() -> None:
    with with_trace_context(
        trace_id="t1",
        span_id="s1",
        parent_span_id="p1",
        session_id="sess1",
        root_request_id="r1",
    ):
        snapshot = get_trace_context()

    assert snapshot == TraceContext(
        trace_id="t1",
        span_id="s1",
        parent_span_id="p1",
        session_id="sess1",
        root_request_id="r1",
    )

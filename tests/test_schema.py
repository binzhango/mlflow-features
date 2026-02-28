import pytest

from agent_mlflow_telemetry.schema import SpanRecord, build_span_attributes


def test_span_record_builds_required_canonical_attributes() -> None:
    span = SpanRecord(
        trace_id="trace-1",
        span_id="span-1",
        component="llm",
        operation="chat.invoke",
        status="ok",
    )

    attrs = build_span_attributes(span)
    assert attrs["trace_id"] == "trace-1"
    assert attrs["span_id"] == "span-1"
    assert attrs["component"] == "llm"
    assert attrs["operation"] == "chat.invoke"
    assert attrs["status"] == "ok"
    assert attrs["name"] == "chat.invoke"


def test_span_record_includes_optional_fields_when_present() -> None:
    span = SpanRecord(
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id="parent-1",
        component="llm",
        operation="chat.invoke",
        status="error",
        session_id="session-1",
        root_request_id="request-1",
        user_id="user-1",
        service_name="agent-svc",
        service_version="1.0.0",
        environment="prod",
        provider="ollama",
        model_name="glm-4.7-flash",
        gateway_route="/gateway/v1",
        endpoint="/chat",
        latency_ms=123.4,
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        streamed=True,
        prompt_text="hello",
        response_text="world",
        prompt_hash="abc123",
        error_type="RuntimeError",
        error_message="boom",
        http_status=500,
        attributes={"custom_key": "custom-value"},
    )

    attrs = span.to_attributes()
    assert attrs["parent_span_id"] == "parent-1"
    assert attrs["session_id"] == "session-1"
    assert attrs["root_request_id"] == "request-1"
    assert attrs["user_id"] == "user-1"
    assert attrs["service.name"] == "agent-svc"
    assert attrs["service.version"] == "1.0.0"
    assert attrs["env"] == "prod"
    assert attrs["provider"] == "ollama"
    assert attrs["model_name"] == "glm-4.7-flash"
    assert attrs["gateway_route"] == "/gateway/v1"
    assert attrs["endpoint"] == "/chat"
    assert attrs["latency_ms"] == 123.4
    assert attrs["input_tokens"] == 20
    assert attrs["output_tokens"] == 10
    assert attrs["total_tokens"] == 30
    assert attrs["streamed"] is True
    assert attrs["prompt_text"] == "hello"
    assert attrs["response_text"] == "world"
    assert attrs["prompt_hash"] == "abc123"
    assert attrs["error_type"] == "RuntimeError"
    assert attrs["error_message"] == "boom"
    assert attrs["http_status"] == 500
    assert attrs["custom_key"] == "custom-value"


def test_span_record_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError, match="trace_id"):
        SpanRecord(trace_id=" ", span_id="s1", component="llm", operation="op", status="ok")

    with pytest.raises(ValueError, match="span_id"):
        SpanRecord(trace_id="t1", span_id="", component="llm", operation="op", status="ok")

    with pytest.raises(ValueError, match="operation"):
        SpanRecord(trace_id="t1", span_id="s1", component="llm", operation=" ", status="ok")


def test_span_record_rejects_invalid_component_and_status() -> None:
    with pytest.raises(ValueError, match="component"):
        SpanRecord(trace_id="t1", span_id="s1", component="unknown", operation="op", status="ok")

    with pytest.raises(ValueError, match="status"):
        SpanRecord(trace_id="t1", span_id="s1", component="llm", operation="op", status="done")


def test_span_record_rejects_negative_latency_and_tokens() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        SpanRecord(
            trace_id="t1",
            span_id="s1",
            component="llm",
            operation="op",
            status="ok",
            latency_ms=-1,
        )

    with pytest.raises(ValueError, match="input_tokens"):
        SpanRecord(
            trace_id="t1",
            span_id="s1",
            component="llm",
            operation="op",
            status="ok",
            input_tokens=-1,
        )

    with pytest.raises(ValueError, match="total_tokens"):
        SpanRecord(
            trace_id="t1",
            span_id="s1",
            component="llm",
            operation="op",
            status="ok",
            total_tokens=-1,
        )


def test_span_record_rejects_total_tokens_mismatch() -> None:
    with pytest.raises(ValueError, match="total_tokens"):
        SpanRecord(
            trace_id="t1",
            span_id="s1",
            component="llm",
            operation="op",
            status="ok",
            input_tokens=1,
            output_tokens=2,
            total_tokens=10,
        )


def test_error_status_requires_error_fields() -> None:
    with pytest.raises(ValueError, match="error status"):
        SpanRecord(
            trace_id="t1",
            span_id="s1",
            component="llm",
            operation="op",
            status="error",
        )


def test_required_attributes_cannot_be_overridden_by_custom_attributes() -> None:
    span = SpanRecord(
        trace_id="t1",
        span_id="s1",
        component="llm",
        operation="op",
        status="ok",
        attributes={"trace_id": "override"},
    )
    with pytest.raises(ValueError, match="required keys"):
        span.to_attributes()

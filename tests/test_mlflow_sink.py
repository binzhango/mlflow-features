from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent_mlflow_telemetry.config import TelemetryConfig
from agent_mlflow_telemetry.mlflow_sink import MLflowSink
from agent_mlflow_telemetry.schema import SpanRecord


@dataclass
class FakeLiveSpan:
    attrs: dict[str, object] = field(default_factory=dict)
    events: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    ended: bool = False
    end_payload: dict[str, object] = field(default_factory=dict)

    def set_attributes(self, attrs: dict[str, object]) -> None:
        self.attrs.update(attrs)

    def add_event(self, event) -> None:
        self.events.append((event.name, dict(event.attributes)))

    def end(self, *, outputs=None, attributes=None, status=None) -> None:
        self.ended = True
        self.end_payload = {
            "outputs": outputs,
            "attributes": attributes,
            "status": status,
        }


class FakeMlflow:
    def __init__(self, *, start_failures: int = 0, active_run_id: str | None = None) -> None:
        self.start_failures = start_failures
        self.active_run_id = active_run_id
        self.start_calls = 0
        self.set_tracking_uri_calls: list[str] = []
        self.set_experiment_calls: list[str] = []
        self.created: list[FakeLiveSpan] = []
        self.start_span_kwargs: list[dict[str, object]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.set_tracking_uri_calls.append(uri)

    def set_experiment(self, exp: str) -> None:
        self.set_experiment_calls.append(exp)

    def start_span_no_context(self, **kwargs):
        self.start_calls += 1
        if self.start_failures > 0:
            self.start_failures -= 1
            raise ConnectionError("temporary unavailable")
        self.start_span_kwargs.append(kwargs)
        span = FakeLiveSpan()
        self.created.append(span)
        return span

    def active_run(self):
        if self.active_run_id is None:
            return None

        @dataclass
        class _Info:
            run_id: str

        @dataclass
        class _Run:
            info: _Info

        return _Run(info=_Info(run_id=self.active_run_id))


def _span(status: str = "ok") -> SpanRecord:
    base = {
        "trace_id": "t1",
        "span_id": "s1",
        "component": "llm",
        "operation": "llm.invoke",
        "status": status,
    }
    if status == "error":
        base["error_type"] = "RuntimeError"
    return SpanRecord(**base)


def test_sink_initializes_tracking_uri_and_experiment() -> None:
    fake = FakeMlflow()
    cfg = TelemetryConfig(mlflow_tracking_uri="http://mlflow:5000", mlflow_experiment="exp-a")
    MLflowSink(cfg, mlflow_module=fake)

    assert fake.set_tracking_uri_calls == ["http://mlflow:5000"]
    assert fake.set_experiment_calls == ["exp-a"]


def test_sink_retries_transient_start_failure_then_succeeds() -> None:
    fake = FakeMlflow(start_failures=1)
    sink = MLflowSink(TelemetryConfig(), mlflow_module=fake, max_retries=2, retry_backoff_seconds=0)

    sink.start_span(_span())

    assert fake.start_calls == 2
    sink.end_span(_span())
    assert fake.created[0].ended is True


def test_sink_fail_open_drops_span_when_retries_exhausted() -> None:
    fake = FakeMlflow(start_failures=10)
    sink = MLflowSink(TelemetryConfig(fail_open=True), mlflow_module=fake, max_retries=1, retry_backoff_seconds=0)

    sink.start_span(_span())
    sink.end_span(_span())

    assert fake.start_calls == 2
    assert fake.created == []


def test_sink_fail_closed_raises_when_retries_exhausted() -> None:
    fake = FakeMlflow(start_failures=10)
    sink = MLflowSink(TelemetryConfig(fail_open=False), mlflow_module=fake, max_retries=1, retry_backoff_seconds=0)

    with pytest.raises(RuntimeError, match="telemetry action failed"):
        sink.start_span(_span())


def test_sink_records_event_on_active_span() -> None:
    fake = FakeMlflow()
    sink = MLflowSink(TelemetryConfig(), mlflow_module=fake)
    sink.start_span(_span())

    sink.record_event("s1", "chunk", {"index": 1})
    sink.end_span(_span())

    assert fake.created[0].events == [("chunk", {"index": 1})]


def test_sink_end_span_sets_error_status() -> None:
    fake = FakeMlflow()
    sink = MLflowSink(TelemetryConfig(), mlflow_module=fake)
    sink.start_span(_span())

    sink.end_span(_span(status="error"))

    assert fake.created[0].ended is True
    # Ensure status object is propagated (enum instance from mlflow)
    assert fake.created[0].end_payload["status"] is not None


def test_sink_populates_trace_metadata_inputs_outputs_and_usage() -> None:
    fake = FakeMlflow()
    sink = MLflowSink(TelemetryConfig(), mlflow_module=fake)
    span = SpanRecord(
        trace_id="t1",
        span_id="s1",
        component="llm",
        operation="llm.invoke",
        status="ok",
        session_id="session-1",
        user_id="user-1",
        model_name="glm-4.7-flash",
        provider="ollama",
        prompt_text="hello",
        response_text="world",
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
    )

    sink.start_span(span)
    sink.end_span(span)

    start_kwargs = fake.start_span_kwargs[0]
    assert start_kwargs["inputs"] == "hello"
    assert start_kwargs["metadata"] == {
        "mlflow.trace.session": "session-1",
        "mlflow.trace.user": "user-1",
    }
    attrs = fake.created[0].attrs
    assert attrs["mlflow.llm.model"] == "glm-4.7-flash"
    assert attrs["mlflow.llm.provider"] == "ollama"
    assert attrs["mlflow.chat.tokenUsage"] == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    assert fake.created[0].end_payload["outputs"] == "world"


def test_sink_links_source_run_when_active_run_exists() -> None:
    fake = FakeMlflow(active_run_id="run-123")
    sink = MLflowSink(TelemetryConfig(), mlflow_module=fake)
    span = SpanRecord(
        trace_id="t1",
        span_id="s1",
        component="llm",
        operation="llm.invoke",
        status="ok",
        prompt_text="hello",
    )

    sink.start_span(span)
    start_kwargs = fake.start_span_kwargs[0]
    assert start_kwargs["metadata"]["mlflow.sourceRun"] == "run-123"


def test_sink_uses_structured_chat_payloads_and_drops_private_attrs() -> None:
    fake = FakeMlflow()
    sink = MLflowSink(TelemetryConfig(), mlflow_module=fake)
    span = SpanRecord(
        trace_id="t1",
        span_id="s1",
        component="llm",
        operation="llm.invoke",
        status="ok",
        attributes={
            "_mlflow_inputs": {"messages": [{"role": "user", "content": "hello"}]},
            "_mlflow_outputs": {
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]
            },
            "mlflow.message.format": "openai",
        },
    )

    sink.start_span(span)
    sink.end_span(span)

    assert fake.start_span_kwargs[0]["inputs"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert fake.created[0].end_payload["outputs"] == {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]
    }
    assert "_mlflow_inputs" not in fake.created[0].attrs
    assert "_mlflow_outputs" not in fake.created[0].attrs

import pytest

from agent_mlflow_telemetry.bootstrap import (
    TelemetryClient,
    TelemetryRuntime,
    build_custom_langchain_tracer,
    build_langchain_callback,
    create_langchain_tracer,
    initialize_telemetry,
)
from agent_mlflow_telemetry.config import TelemetryConfig
from agent_mlflow_telemetry.langchain_callback import CustomLangchainTracer
from agent_mlflow_telemetry.mlflow_sink import MLflowSink
from agent_mlflow_telemetry.schema import SpanRecord


def test_telemetry_config_defaults() -> None:
    cfg = TelemetryConfig()

    assert cfg.enabled is True
    assert cfg.mlflow_tracking_uri is None
    assert cfg.mlflow_experiment is None
    assert cfg.sampling_rate == 1.0
    assert cfg.log_content is True
    assert cfg.fail_open is True
    assert cfg.service_name == "mlflow-features"
    assert cfg.service_version is None
    assert cfg.environment == "dev"


def test_telemetry_config_validates_constraints() -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        TelemetryConfig(sampling_rate=1.5)

    with pytest.raises(ValueError, match="service_name"):
        TelemetryConfig(service_name="  ")

    with pytest.raises(ValueError, match="environment"):
        TelemetryConfig(environment="")


def test_telemetry_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("AGENT_TELEMETRY_MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("AGENT_TELEMETRY_MLFLOW_EXPERIMENT", "agent-exp")
    monkeypatch.setenv("AGENT_TELEMETRY_SAMPLING_RATE", "0.25")
    monkeypatch.setenv("AGENT_TELEMETRY_LOG_CONTENT", "0")
    monkeypatch.setenv("AGENT_TELEMETRY_FAIL_OPEN", "yes")
    monkeypatch.setenv("AGENT_TELEMETRY_SERVICE_NAME", "svc-a")
    monkeypatch.setenv("AGENT_TELEMETRY_SERVICE_VERSION", "1.2.3")
    monkeypatch.setenv("AGENT_TELEMETRY_ENVIRONMENT", "prod")

    cfg = TelemetryConfig.from_env()
    assert cfg.enabled is False
    assert cfg.mlflow_tracking_uri == "http://mlflow:5000"
    assert cfg.mlflow_experiment == "agent-exp"
    assert cfg.sampling_rate == 0.25
    assert cfg.log_content is False
    assert cfg.fail_open is True
    assert cfg.service_name == "svc-a"
    assert cfg.service_version == "1.2.3"
    assert cfg.environment == "prod"


def test_telemetry_config_as_dict() -> None:
    cfg = TelemetryConfig(service_name="svc", environment="test")
    payload = cfg.as_dict()

    assert payload["service_name"] == "svc"
    assert payload["environment"] == "test"
    assert payload["sampling_rate"] == 1.0


def test_initialize_runtime_and_build_callback_wires_sink() -> None:
    cfg = TelemetryConfig(service_name="svc", environment="test")
    runtime = initialize_telemetry(cfg)

    assert isinstance(runtime, TelemetryRuntime)
    assert isinstance(runtime, TelemetryClient)
    assert runtime.config is cfg
    assert isinstance(runtime.sink, MLflowSink)
    assert runtime.sink._config is cfg

    cb = build_langchain_callback(runtime)
    assert isinstance(cb, CustomLangchainTracer)
    assert cb._sink is runtime.sink

    canonical = create_langchain_tracer(runtime)
    assert isinstance(canonical, CustomLangchainTracer)
    assert canonical._sink is runtime.sink

    tracer = build_custom_langchain_tracer(runtime)
    assert isinstance(tracer, CustomLangchainTracer)
    assert tracer._sink is runtime.sink


def test_sink_noop_methods_do_not_raise() -> None:
    cfg = TelemetryConfig(enabled=False)
    sink = MLflowSink(cfg)
    span = SpanRecord(
        trace_id="t1",
        span_id="s1",
        component="llm",
        operation="invoke",
        status="ok",
        name="n1",
    )

    assert sink.start_span(span) is None
    assert sink.end_span(span) is None
    assert sink.record_event("s1", "evt", {"k": "v"}) is None

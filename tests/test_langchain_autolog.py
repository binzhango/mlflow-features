from __future__ import annotations

from langchain_core.callbacks import CallbackManager

from agent_mlflow_telemetry.bootstrap import TelemetryRuntime
from agent_mlflow_telemetry.config import TelemetryConfig
from agent_mlflow_telemetry.langchain_autolog import (
    autolog,
    disable_autolog,
    enable_autolog,
    reset_autolog_state,
)
from agent_mlflow_telemetry.langchain_callback import CustomLangchainTracer
from agent_mlflow_telemetry.mlflow_sink import MLflowSink


def setup_function() -> None:
    reset_autolog_state()


def teardown_function() -> None:
    reset_autolog_state()


def test_autolog_builds_runtime_and_auto_injects_tracer() -> None:
    runtime = autolog(config=TelemetryConfig(enabled=False))
    assert isinstance(runtime, TelemetryRuntime)

    manager = CallbackManager(handlers=[])
    tracers = [h for h in manager.inheritable_handlers if isinstance(h, CustomLangchainTracer)]
    assert len(tracers) == 1
    assert tracers[0].run_inline is True


def test_enable_autolog_uses_provided_runtime() -> None:
    runtime = TelemetryRuntime(
        config=TelemetryConfig(enabled=False, service_name="svc-a"),
        sink=MLflowSink(TelemetryConfig(enabled=False, service_name="svc-a")),
    )
    enable_autolog(runtime)

    manager = CallbackManager(handlers=[])
    tracer = [h for h in manager.inheritable_handlers if isinstance(h, CustomLangchainTracer)][0]
    assert tracer._sink is runtime.sink
    assert tracer.run_inline is True


def test_autolog_can_disable_inline_tracer_execution() -> None:
    _ = autolog(config=TelemetryConfig(enabled=False), run_tracer_inline=False)

    manager = CallbackManager(handlers=[])
    tracer = [h for h in manager.inheritable_handlers if isinstance(h, CustomLangchainTracer)][0]
    assert tracer.run_inline is False


def test_disable_autolog_stops_injection() -> None:
    _ = autolog(config=TelemetryConfig(enabled=False))
    disable_autolog()

    manager = CallbackManager(handlers=[])
    tracers = [h for h in manager.inheritable_handlers if isinstance(h, CustomLangchainTracer)]
    assert tracers == []


def test_autolog_merge_has_single_tracer() -> None:
    _ = autolog(config=TelemetryConfig(enabled=False))

    left = CallbackManager(handlers=[])
    right = CallbackManager(handlers=[])
    merged = left.merge(right)

    tracers = [h for h in merged.inheritable_handlers if isinstance(h, CustomLangchainTracer)]
    assert len(tracers) == 1

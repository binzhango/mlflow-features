import agent_mlflow_telemetry as telemetry
from agent_mlflow_telemetry.bootstrap import (
    build_custom_langchain_tracer,
    build_langchain_callback,
    initialize_telemetry,
)
from agent_mlflow_telemetry.config import TelemetryConfig
from agent_mlflow_telemetry.context import with_trace_context
from agent_mlflow_telemetry.langchain_autolog import autolog, disable_autolog, enable_autolog
from agent_mlflow_telemetry.langchain_callback import CustomLangchainTracer


def test_public_exports_are_present_and_ordered() -> None:
    assert telemetry.__all__ == [
        "TelemetryConfig",
        "initialize_telemetry",
        "build_langchain_callback",
        "build_custom_langchain_tracer",
        "CustomLangchainTracer",
        "autolog",
        "enable_autolog",
        "disable_autolog",
        "with_trace_context",
        "wrap_llmclient",
    ]


def test_public_exports_reference_expected_symbols() -> None:
    assert telemetry.TelemetryConfig is TelemetryConfig
    assert telemetry.initialize_telemetry is initialize_telemetry
    assert telemetry.build_langchain_callback is build_langchain_callback
    assert telemetry.build_custom_langchain_tracer is build_custom_langchain_tracer
    assert telemetry.CustomLangchainTracer is CustomLangchainTracer
    assert telemetry.autolog is autolog
    assert telemetry.enable_autolog is enable_autolog
    assert telemetry.disable_autolog is disable_autolog
    assert telemetry.with_trace_context is with_trace_context
    assert callable(telemetry.wrap_llmclient)

import agent_mlflow_telemetry as telemetry
from agent_mlflow_telemetry.bootstrap import (
    TelemetryClient,
    build_custom_langchain_tracer,
    build_langchain_callback,
    create_langchain_tracer,
    initialize_telemetry,
)
from agent_mlflow_telemetry.config import TelemetryConfig
from agent_mlflow_telemetry.context import with_trace_context
from agent_mlflow_telemetry.langchain_autolog import autolog, disable_autolog, enable_autolog
from agent_mlflow_telemetry.langchain_callback import CustomLangchainTracer, LangChainTelemetryTracer


def test_public_exports_are_present_and_ordered() -> None:
    assert telemetry.__all__ == [
        "TelemetryConfig",
        "TelemetryClient",
        "initialize_telemetry",
        "create_langchain_tracer",
        "build_langchain_callback",
        "build_custom_langchain_tracer",
        "LangChainTelemetryTracer",
        "CustomLangchainTracer",
        "autolog",
        "enable_autolog",
        "disable_autolog",
        "with_trace_context",
        "TelemetryEnabledClient",
        "create_telemetry_client",
        "wrap_llmclient",
    ]


def test_public_exports_reference_expected_symbols() -> None:
    assert telemetry.TelemetryConfig is TelemetryConfig
    assert telemetry.TelemetryClient is TelemetryClient
    assert telemetry.initialize_telemetry is initialize_telemetry
    assert telemetry.create_langchain_tracer is create_langchain_tracer
    assert telemetry.build_langchain_callback is build_langchain_callback
    assert telemetry.build_custom_langchain_tracer is build_custom_langchain_tracer
    assert telemetry.LangChainTelemetryTracer is LangChainTelemetryTracer
    assert telemetry.CustomLangchainTracer is CustomLangchainTracer
    assert telemetry.autolog is autolog
    assert telemetry.enable_autolog is enable_autolog
    assert telemetry.disable_autolog is disable_autolog
    assert telemetry.with_trace_context is with_trace_context
    assert callable(telemetry.create_telemetry_client)
    assert callable(telemetry.wrap_llmclient)

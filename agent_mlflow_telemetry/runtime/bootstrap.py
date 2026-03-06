"""Public bootstrap helpers for telemetry setup."""

from dataclasses import dataclass

from .config import TelemetryConfig
from ..integrations.langchain.tracer import LangChainTelemetryTracer
from ..sinks.mlflow import MLflowSink


@dataclass(slots=True)
class TelemetryClient:
    """Runtime container shared by callbacks and wrappers."""

    config: TelemetryConfig
    sink: MLflowSink


def initialize_telemetry(config: TelemetryConfig) -> TelemetryClient:
    """Initialize telemetry runtime and sink."""

    sink = MLflowSink(config=config)
    return TelemetryClient(config=config, sink=sink)


def create_langchain_tracer(runtime: TelemetryClient) -> LangChainTelemetryTracer:
    """Create LangChain telemetry tracer from runtime."""
    return LangChainTelemetryTracer(sink=runtime.sink)


def build_langchain_callback(runtime: TelemetryClient) -> LangChainTelemetryTracer:
    """Backward-compatible alias for create_langchain_tracer()."""
    return create_langchain_tracer(runtime)


def build_custom_langchain_tracer(runtime: TelemetryClient) -> LangChainTelemetryTracer:
    """Backward-compatible alias for create_langchain_tracer()."""
    return create_langchain_tracer(runtime)


TelemetryRuntime = TelemetryClient

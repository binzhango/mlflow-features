"""Public bootstrap helpers for telemetry setup."""

from dataclasses import dataclass

from .config import TelemetryConfig
from .langchain_callback import CustomLangchainTracer
from .mlflow_sink import MLflowSink


@dataclass(slots=True)
class TelemetryRuntime:
    """Runtime container shared by callbacks and wrappers."""

    config: TelemetryConfig
    sink: MLflowSink


def initialize_telemetry(config: TelemetryConfig) -> TelemetryRuntime:
    """Initialize telemetry runtime and sink."""

    sink = MLflowSink(config=config)
    return TelemetryRuntime(config=config, sink=sink)


def build_langchain_callback(runtime: TelemetryRuntime) -> CustomLangchainTracer:
    """Build LangChain callback handler from runtime."""

    return CustomLangchainTracer(sink=runtime.sink)


def build_custom_langchain_tracer(runtime: TelemetryRuntime) -> CustomLangchainTracer:
    """Build custom tracer callback handler from runtime."""

    return CustomLangchainTracer(sink=runtime.sink)

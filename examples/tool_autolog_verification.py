"""Verify tool-call tracing in autolog mode with ChatOllama.

Run:
    UV_CACHE_DIR=.uv-cache uv run python examples/tool_autolog_verification.py

Optional env:
    MLFLOW_TRACKING_URI
    AGENT_TELEMETRY_MLFLOW_TRACKING_URI
    MLFLOW_EXPERIMENT
    OLLAMA_MODEL
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import mlflow
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Allow direct script execution via:
# `uv run python examples/tool_autolog_verification.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_mlflow_telemetry import TelemetryConfig, autolog, disable_autolog  # noqa: E402


def _tracking_uri() -> str | None:
    return os.getenv("AGENT_TELEMETRY_MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")


def _search_trace_for_prompt(*, experiment_id: str, prompt: str, attempts: int = 8, sleep_seconds: float = 0.25):
    def _trace_has_prompt(trace: object) -> bool:
        if prompt in str(getattr(trace.info, "request_preview", "")):
            return True
        for span in trace.search_spans():
            if prompt in str(getattr(span, "inputs", "")):
                return True
        return False

    for _ in range(attempts):
        traces = mlflow.search_traces(
            locations=[experiment_id],
            max_results=100,
            order_by=["timestamp_ms DESC"],
            include_spans=True,
            return_type="list",
        )
        for trace in traces:
            if _trace_has_prompt(trace):
                return trace
        time.sleep(sleep_seconds)
    return None


@tool
def lookup_weather(city: str) -> str:
    """Lookup a city's weather."""
    return f"{city}: 21C and clear"


def main() -> int:
    tracking_uri = _tracking_uri()
    experiment = os.getenv("MLFLOW_EXPERIMENT", "agent-telemetry-tool-autolog-verify")
    model_name = os.getenv("OLLAMA_MODEL", "nemotron-3-nano")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment)
    experiment_id = exp.experiment_id

    cfg = TelemetryConfig(
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=experiment,
        service_name="tool-autolog-verification-example",
        environment=os.getenv("ENVIRONMENT", "dev"),
    )

    if autolog(config=cfg) is None:
        print("Failed to enable autolog.")
        return 1

    model = ChatOllama(model=model_name, temperature=0)
    model_with_tools = model.bind_tools([lookup_weather])
    marker = uuid4().hex[:8]
    prompt = (
        f"[tool-autolog-{marker}] You MUST call lookup_weather once for city=Boston. "
        "Return only the tool call."
    )

    try:
        # No callbacks and no mlflow.start_run(): verify pure autolog flow.
        response = model_with_tools.invoke(prompt)
        tool_calls = getattr(response, "tool_calls", None) or []
        print("model response content:", getattr(response, "content", ""))
        print("model tool_calls:", tool_calls)

        trace = _search_trace_for_prompt(experiment_id=experiment_id, prompt=prompt)
        if trace is None:
            print("Verification failed: no trace found for prompt marker.")
            return 1

        tool_call_span_names = [span.name for span in trace.search_spans() if span.name.startswith("tool_call.")]
        if not tool_call_span_names:
            print("Verification failed: trace found but no tool_call.* child spans.")
            print(f"trace_id: {trace.info.trace_id}")
            return 1

        print("Tool autolog verification passed.")
        print(f"trace_id: {trace.info.trace_id}")
        print("tool_call spans:", tool_call_span_names)
        return 0
    except Exception as exc:  # noqa: BLE001
        print("Tool autolog verification failed.")
        print(
            "Ensure Ollama is running, model supports tool calling, and model is available "
            f"(model={model_name!r})."
        )
        print(f"Error: {exc}")
        return 1
    finally:
        disable_autolog()


if __name__ == "__main__":
    raise SystemExit(main())

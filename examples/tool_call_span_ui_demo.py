"""Deterministic tool-call telemetry demo for MLflow trace UI.

Run:
    ./scripts/run_mlflow_server.sh
    AGENT_TELEMETRY_MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
    UV_CACHE_DIR=.uv-cache uv run python examples/tool_call_span_ui_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from uuid import uuid4

import mlflow

# Allow direct script execution via:
# `uv run python examples/tool_call_span_ui_demo.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_mlflow_telemetry import (  # noqa: E402
    TelemetryConfig,
    build_langchain_callback,
    initialize_telemetry,
    with_trace_context,
)
from agent_mlflow_telemetry.langchain_callback import TelemetryCallbackHandler  # noqa: E402


@dataclass
class _Msg:
    content: str
    tool_calls: list[dict[str, object]] | None = None
    response_metadata: dict[str, object] | None = None
    additional_kwargs: dict[str, object] | None = None
    usage_metadata: dict[str, int] | None = None


@dataclass
class _Gen:
    text: str = ""
    message: _Msg | None = None


@dataclass
class _Resp:
    generations: list[list[_Gen]]
    llm_output: dict[str, object]


def _tracking_uri() -> str | None:
    return os.getenv("AGENT_TELEMETRY_MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")


def _emit_trace(*, callback: TelemetryCallbackHandler, question: str) -> None:
    chain_run_id = uuid4()
    llm_plan_run_id = uuid4()
    tool_exec_run_id = uuid4()
    llm_answer_run_id = uuid4()

    callback.on_chain_start(
        {"name": "tool_call_demo_chain"},
        {"question": question},
        run_id=chain_run_id,
        parent_run_id=None,
        metadata={"user_id": os.getenv("USER", "unknown"), "session_id": "tool-call-ui-demo"},
    )

    callback.on_llm_start(
        {"name": "FakeToolPlanner"},
        [question],
        run_id=llm_plan_run_id,
        parent_run_id=chain_run_id,
        invocation_params={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.1,
            "reasoning": {"effort": "high"},
        },
    )
    callback.on_llm_end(
        _Resp(
            generations=[
                [
                    _Gen(
                        message=_Msg(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call_weather_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": "{\"city\":\"Boston\"}",
                                    },
                                }
                            ],
                            response_metadata={"finish_reason": "tool_calls"},
                        )
                    )
                ]
            ],
            llm_output={"token_usage": {"prompt_tokens": 26, "completion_tokens": 8, "total_tokens": 34}},
        ),
        run_id=llm_plan_run_id,
        parent_run_id=chain_run_id,
    )

    callback.on_tool_start(
        {"name": "lookup_weather"},
        "{\"city\":\"Boston\"}",
        run_id=tool_exec_run_id,
        parent_run_id=chain_run_id,
    )
    callback.on_tool_end(
        {"city": "Boston", "temp_c": 6, "condition": "cloudy"},
        run_id=tool_exec_run_id,
        parent_run_id=chain_run_id,
    )

    callback.on_llm_start(
        {"name": "FakeToolResponder"},
        [question],
        run_id=llm_answer_run_id,
        parent_run_id=chain_run_id,
        invocation_params={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.3,
            "reasoning_effort": "medium",
        },
    )
    callback.on_llm_end(
        _Resp(
            generations=[[_Gen(text="In Boston it is 6C and cloudy.")]],
            llm_output={"token_usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}},
        ),
        run_id=llm_answer_run_id,
        parent_run_id=chain_run_id,
    )

    callback.on_chain_end(
        "In Boston it is 6C and cloudy.",
        run_id=chain_run_id,
        parent_run_id=None,
    )


def _verify_tool_call_spans(*, run_id: str, root_request_id: str) -> list[str]:
    traces = mlflow.search_traces(
        run_id=run_id,
        max_results=20,
        order_by=["timestamp_ms DESC"],
        include_spans=True,
        return_type="list",
    )
    if not traces:
        return []

    chosen = next(
        (
            trace
            for trace in traces
            if getattr(trace.info, "client_request_id", None) == root_request_id
        ),
        traces[0],
    )
    return [span.name for span in chosen.search_spans() if span.name.startswith("tool_call.")]


def main() -> int:
    tracking_uri = _tracking_uri()
    experiment = os.getenv("MLFLOW_EXPERIMENT", "agent-telemetry-demo")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    cfg = TelemetryConfig(
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=experiment,
        service_name="tool-call-span-ui-demo",
        environment=os.getenv("ENVIRONMENT", "dev"),
    )
    runtime = initialize_telemetry(cfg)
    callback = build_langchain_callback(runtime)

    root_request_id = f"tool-call-ui-demo-{uuid4().hex[:8]}"
    question = "What is the weather in Boston right now?"

    try:
        with mlflow.start_run(run_name="tool-call-span-ui-demo") as run:
            with with_trace_context(
                session_id="tool-call-ui-demo-session",
                root_request_id=root_request_id,
            ):
                _emit_trace(callback=callback, question=question)

            tool_call_spans = _verify_tool_call_spans(run_id=run.info.run_id, root_request_id=root_request_id)
            print(f"run_id: {run.info.run_id}")
            print(f"experiment: {experiment}")
            print(f"root_request_id: {root_request_id}")
            if tool_call_spans:
                print("tool_call spans found:")
                for name in tool_call_spans:
                    print(f"  - {name}")
            else:
                print("No tool_call spans found via search API. Check MLflow server connectivity.")
    except Exception as exc:  # noqa: BLE001
        print("Tool-call demo failed.")
        if tracking_uri:
            print(f"Tracking URI: {tracking_uri}")
        print("Ensure MLflow tracking is reachable and try again.")
        print(f"Error: {exc}")
        return 1

    print("UI verification:")
    print("1) Open MLflow UI and select the latest trace from run 'tool-call-span-ui-demo'.")
    print("2) Expand child spans under the first LLM span.")
    print("3) Confirm a span named 'tool_call.function.lookup_weather' exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

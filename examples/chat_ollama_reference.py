"""Reference telemetry integration with ChatOllama (glm-4.7-flash).

Run:
    UV_CACHE_DIR=.uv-cache uv run python examples/chat_ollama_reference.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import mlflow

# Allow direct script execution via:
# `uv run python examples/chat_ollama_reference.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_ollama import ChatOllama

from agent_mlflow_telemetry import (
    TelemetryConfig,
    build_langchain_callback,
    initialize_telemetry,
    wrap_llmclient,
)
from agent_mlflow_telemetry.context import with_trace_context


def main() -> None:
    cfg = TelemetryConfig(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT", "agent-telemetry-demo"),
        service_name="chat-ollama-reference",
        environment=os.getenv("ENVIRONMENT", "dev"),
    )
    runtime = initialize_telemetry(cfg)
    callback = build_langchain_callback(runtime)

    model = ChatOllama(model="nemotron-3-nano", temperature=0)
    wrapped = wrap_llmclient(model, runtime.sink)

    try:
        # Start an MLflow run so trace rows can resolve the "Run name" column.
        with mlflow.start_run(run_name="chat-ollama-telemetry-demo"):
            with with_trace_context(
                session_id=os.getenv("AGENT_TELEMETRY_SESSION_ID", "demo-session"),
                root_request_id=os.getenv("AGENT_TELEMETRY_ROOT_REQUEST_ID", "demo-root-request"),
            ):
                # Callback-based tracing on LangChain model invocation.
                response = model.invoke(
                    "Reply with exactly: telemetry-ready",
                    config={"callbacks": [callback]},
                )
                print("callback invoke response:", getattr(response, "content", response))

                # Wrapper-based tracing on ChatModel-compatible client.
                wrapped_response = wrapped.invoke(
                    "Say hello in 3 words",
                    config={
                        "metadata": {
                            "session_id": os.getenv("AGENT_TELEMETRY_SESSION_ID", "demo-session"),
                            "user_id": os.getenv("USER", "unknown"),
                        }
                    },
                )
                print("wrapped invoke response:", getattr(wrapped_response, "content", wrapped_response))
    except Exception as exc:  # noqa: BLE001
        print("Example run failed. Ensure Ollama is running and model glm-4.7-flash is available.")
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()

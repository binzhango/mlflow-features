"""Verify autolog-only tracing with ChatOllama.

Run:
    UV_CACHE_DIR=.uv-cache uv run python examples/autolog_verification.py

Optional env:
    MLFLOW_TRACKING_URI
    AGENT_TELEMETRY_MLFLOW_TRACKING_URI
    MLFLOW_EXPERIMENT
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import mlflow
from langchain_ollama import ChatOllama

# Allow direct script execution via:
# `uv run python examples/autolog_verification.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_mlflow_telemetry import TelemetryConfig, autolog, disable_autolog  # noqa: E402


def _tracking_uri() -> str | None:
    return os.getenv("AGENT_TELEMETRY_MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")


def _span_type_text(span_type: object) -> str:
    name = getattr(span_type, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(span_type)


def _search_traces_by_prompt(
    *,
    experiment_id: str,
    prompts: set[str],
    attempts: int = 8,
    sleep_seconds: float = 0.25,
) -> dict[str, object]:
    found: dict[str, object] = {}

    def _contains_prompt(trace: object, prompt: str) -> bool:
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
        for prompt in prompts:
            if prompt in found:
                continue
            for trace in traces:
                if _contains_prompt(trace, prompt):
                    found[prompt] = trace
                    break
        if prompts.issubset(found.keys()):
            return found
        time.sleep(sleep_seconds)
    return found


def main() -> int:
    tracking_uri = _tracking_uri()
    experiment = os.getenv("MLFLOW_EXPERIMENT", "agent-telemetry-autolog-verify")
    model_name = os.getenv("OLLAMA_MODEL", "nemotron-3-nano")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment)
    experiment_id = exp.experiment_id

    cfg = TelemetryConfig(
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=experiment,
        service_name="autolog-verification-example",
        environment=os.getenv("ENVIRONMENT", "dev"),
    )

    if autolog(config=cfg) is None:
        print("Failed to enable autolog.")
        return 1

    model = ChatOllama(model=model_name, temperature=0)
    try:
        sync_marker = uuid4().hex[:8]
        async_marker = uuid4().hex[:8]
        sync_prompt = f"Reply with exactly: telemetry-ready sync-{sync_marker}"
        async_prompt = f"Reply with exactly: telemetry-ready async-{async_marker}"

        # No callbacks and no mlflow.start_run() on purpose: this validates autolog path.
        sync_response = model.invoke(sync_prompt)
        print("invoke response:", getattr(sync_response, "content", sync_response))

        async_response = asyncio.run(model.ainvoke(async_prompt))
        print("ainvoke response:", getattr(async_response, "content", async_response))

        traces_by_prompt = _search_traces_by_prompt(
            experiment_id=experiment_id,
            prompts={sync_prompt, async_prompt},
        )
        missing = {sync_prompt, async_prompt}.difference(traces_by_prompt.keys())
        if missing:
            print("Verification failed: missing traces for prompts:")
            for prompt in sorted(missing):
                print(f"  - {prompt}")
            return 1

        for mode, prompt in (("invoke", sync_prompt), ("ainvoke", async_prompt)):
            trace = traces_by_prompt[prompt]
            spans = trace.search_spans()
            chat_spans = [
                span
                for span in spans
                if "CHAT_MODEL" in _span_type_text(getattr(span, "span_type", ""))
            ]
            if not chat_spans:
                print(f"Verification failed: trace has no CHAT_MODEL span ({mode}).")
                print(f"trace_id: {trace.info.trace_id}")
                print("span names:", [span.name for span in spans])
                return 1

            print(f"{mode} trace verified: trace_id={trace.info.trace_id}")
            print("chat spans:", [span.name for span in chat_spans])

        print("Autolog check passed (invoke + ainvoke).")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("Autolog verification failed.")
        print(
            "Ensure Ollama is running and the model is available "
            f"(model={model_name!r})."
        )
        print(f"Error: {exc}")
        return 1
    finally:
        disable_autolog()


if __name__ == "__main__":
    raise SystemExit(main())

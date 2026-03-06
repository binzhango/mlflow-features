"""Verify explicit open_trace_context() / close_trace_context() behavior."""

from __future__ import annotations

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.manager import CallbackManager
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import (
    close_trace_context,
    enable_mlflow_langchain_enrichment,
    get_current_trace_context,
    open_trace_context,
)
from mlflow_langchain_enrichment.enrichment import TraceEnrichmentCallback


def _print_state(label: str) -> None:
    current = get_current_trace_context()
    active_run = mlflow.active_run()
    print(f"\n[{label}]")
    print(f"trace_context: {current}")
    print(f"active_run: {getattr(getattr(active_run, 'info', None), 'run_id', None)}")


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()
    enable_mlflow_langchain_enrichment()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise support assistant."),
            ("human", "{question}"),
        ]
    )
    chain = (prompt | ChatOllama(model="nemotron-3-nano", temperature=0)).with_config(
        {"run_name": "support-assistant"}
    )

    _print_state("before open")

    handle = open_trace_context(
        user_id="user-42",
        session_id="session-20260306-open-close-001",
        client_request_id="req-20260306-open-close-abc",
        mlflow_run_name="verify-open-close-run",
        ensure_run=True,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "open-close-verification",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
        },
        run_tags={"team": "support"},
        span_metadata={"tenant": "local-demo", "provider": "ollama"},
        trace_name="verify-open-close-trace",
    )

    try:
        _print_state("after open")

        manager = CallbackManager.configure()
        has_trace_handler = any(
            isinstance(handler, TraceEnrichmentCallback) for handler in manager.handlers
        )
        print(f"trace_enrichment_callback_injected: {has_trace_handler}")

        result = chain.invoke({"question": "Why was invoice INV-42 charged twice?"})
        print("\nmodel_response:")
        print(result.content)
    finally:
        close_trace_context(handle)

    _print_state("after close")
    print("\nExpected:")
    print("- after open: trace_context should not be None")
    print("- after open: active_run should have a run id")
    print("- trace_enrichment_callback_injected: True")
    print("- after close: trace_context should be None")
    print("- after close: active_run should be None")


if __name__ == "__main__":
    main()

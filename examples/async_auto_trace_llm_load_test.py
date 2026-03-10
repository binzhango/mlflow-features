"""Verify 10 concurrent async requests using auto_trace_llm(...)."""

from __future__ import annotations

import asyncio
import time

import mlflow
from mlflow.entities.trace_location import MlflowExperimentLocation
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import auto_trace_llm


def build_messages(question: str):
    return [
        SystemMessage(content="You are a concise support assistant."),
        HumanMessage(content=question),
    ]


def build_base_llm():
    return ChatOllama(model="nemotron-3-nano", temperature=0).with_config(
        {"run_name": "support-assistant-auto-trace-load"}
    )


async def run_request(
    *,
    request_id: int,
    session_id: str,
    experiment_id: str,
    started_at: float,
) -> dict[str, str]:
    mlflow.tracing.set_destination(
        MlflowExperimentLocation(experiment_id=experiment_id),
        context_local=True,
    )

    base_llm = build_base_llm()
    traced_llm = auto_trace_llm(
        base_llm,
        user_id=f"user-{request_id:02d}",
        session_id=session_id,
        client_request_id=f"req-20260310-auto-load-{request_id:02d}",
        mlflow_run_name=f"auto-trace-load-{session_id}-{request_id:02d}",
        ensure_run=False,
        capture_root_span_io=True,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "async-auto-trace-load",
            "session_group": session_id,
            "model": getattr(base_llm, "model", "unknown"),
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "request_number": str(request_id),
            "model_name": getattr(base_llm, "model", "unknown"),
        },
        trace_name=f"auto-trace-load-{session_id}",
        metadata_builder=lambda runnable, inputs, response, error: {
            "response_chars": str(len(getattr(response, "content", "") or "")),
        },
    )

    messages = build_messages(
        (
            f"Request {request_id:02d}: summarize why invoice INV-{request_id:02d} "
            f"might be double charged for session {session_id}."
        )
    )

    print(
        f"{time.perf_counter() - started_at:6.2f}s "
        f"start request={request_id:02d} session={session_id}"
    )
    result = await traced_llm.ainvoke(messages)
    print(
        f"{time.perf_counter() - started_at:6.2f}s "
        f"done  request={request_id:02d} session={session_id}"
    )

    return {
        "request_id": f"{request_id:02d}",
        "session_id": session_id,
        "response": result.content,
    }


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    session_id = "session-auto-trace-load-shared"
    started_at = time.perf_counter()

    tasks = [
        asyncio.create_task(
            run_request(
                request_id=index + 1,
                session_id=session_id,
                experiment_id=experiment.experiment_id,
                started_at=started_at,
            )
        )
        for index in range(10)
    ]
    results = await asyncio.gather(*tasks)

    print(f"\nTotal elapsed: {time.perf_counter() - started_at:.2f}s")
    print("\nSummary:")
    for item in results:
        print(
            f"request={item['request_id']} session={item['session_id']} "
            f"response={item['response'][:80]}"
        )


if __name__ == "__main__":
    asyncio.run(main())

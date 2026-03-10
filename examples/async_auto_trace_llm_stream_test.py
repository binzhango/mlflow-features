"""Verify concurrent async streaming requests using auto_trace_llm(...)."""

from __future__ import annotations

import asyncio
import time

import mlflow
from mlflow.entities.trace_location import MlflowExperimentLocation
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import auto_trace_llm


CONCURRENCY = 3
SHARED_SESSION_ID = "session-auto-trace-stream-shared"


def build_messages(question: str):
    return [
        SystemMessage(content="You are a concise support assistant."),
        HumanMessage(content=question),
    ]


def build_base_llm():
    return ChatOllama(model="nemotron-3-nano", temperature=0).with_config(
        {"run_name": "support-assistant-auto-trace-stream"}
    )


async def run_stream_request(
    *,
    request_id: int,
    experiment_id: str,
    started_at: float,
    print_lock: asyncio.Lock,
) -> dict[str, str]:
    mlflow.tracing.set_destination(
        MlflowExperimentLocation(experiment_id=experiment_id),
        context_local=True,
    )

    base_llm = build_base_llm()
    traced_llm = auto_trace_llm(
        base_llm,
        user_id=f"user-{request_id:02d}",
        session_id=SHARED_SESSION_ID,
        client_request_id=f"req-20260310-auto-stream-{request_id:02d}",
        mlflow_run_name=f"auto-trace-stream-{request_id:02d}",
        ensure_run=False,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "async-auto-trace-stream",
            "model": getattr(base_llm, "model", "unknown"),
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "request_number": str(request_id),
            "model_name": getattr(base_llm, "model", "unknown"),
        },
        trace_name=f"auto-trace-stream-{request_id:02d}",
        metadata_builder=lambda runnable, inputs, response, error: {
            "response_chars": str(len(getattr(response, "content", "") or "")),
        },
    )

    messages = build_messages(
        (
            f"Request {request_id:02d}: stream a one-sentence explanation for why invoice "
            f"INV-{request_id:02d} might be double charged."
        )
    )

    async with print_lock:
        print(
            f"{time.perf_counter() - started_at:6.2f}s "
            f"start stream request={request_id:02d}"
        )

    chunks: list[str] = []
    async for chunk in traced_llm.astream(messages):
        text = getattr(chunk, "content", "")
        if text:
            chunks.append(text)
            async with print_lock:
                print(
                    f"{time.perf_counter() - started_at:6.2f}s "
                    f"chunk request={request_id:02d} text={text!r}"
                )

    final_response = "".join(chunks)
    async with print_lock:
        print(
            f"{time.perf_counter() - started_at:6.2f}s "
            f"done  stream request={request_id:02d}"
        )

    return {
        "request_id": f"{request_id:02d}",
        "session_id": SHARED_SESSION_ID,
        "response": final_response,
    }


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    started_at = time.perf_counter()
    print_lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(
            run_stream_request(
                request_id=index + 1,
                experiment_id=experiment.experiment_id,
                started_at=started_at,
                print_lock=print_lock,
            )
        )
        for index in range(CONCURRENCY)
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

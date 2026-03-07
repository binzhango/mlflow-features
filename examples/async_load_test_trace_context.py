"""Verify async trace-context isolation with concurrent LangChain requests."""

from __future__ import annotations

import asyncio

import mlflow
from mlflow.entities.trace_location import MlflowExperimentLocation
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import (
    close_trace_context,
    enable_mlflow_langchain_enrichment,
    get_current_trace_context,
    open_trace_context,
)


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise support assistant."),
            ("human", "{question}"),
        ]
    )
    return (prompt | ChatOllama(model="nemotron-3-nano", temperature=0)).with_config(
        {"run_name": "support-assistant-async-load"}
    )


async def run_request(chain, request_id: int, session_id: str, experiment_id: str) -> dict[str, str]:
    mlflow.tracing.set_destination(
        MlflowExperimentLocation(experiment_id=experiment_id),
        context_local=True,
    )
    handle = open_trace_context(
        user_id=f"user-{request_id:02d}",
        session_id=session_id,
        client_request_id=f"req-20260306-async-{request_id:02d}",
        mlflow_run_name=f"async-load-{session_id}-{request_id:02d}",
        ensure_run=True,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "async-load-test",
            "session_group": session_id,
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "request_number": str(request_id),
        },
        run_tags={"team": "support", "scenario": "async-load-test"},
        span_metadata={"tenant": "local-demo", "provider": "ollama"},
        trace_name=f"async-load-{session_id}",
    )

    try:
        current = get_current_trace_context()
        print(
            f"start request={request_id:02d} session={session_id} "
            f"context_session={current.session_id}"
        )

        result = await chain.ainvoke(
            {
                "question": (
                    f"Request {request_id:02d}: summarize why invoice INV-{request_id:02d} "
                    f"might be double charged for session {session_id}."
                )
            }
        )

        current = get_current_trace_context()
        print(
            f"done  request={request_id:02d} session={session_id} "
            f"context_session={current.session_id}"
        )

        return {
            "request_id": f"{request_id:02d}",
            "session_id": session_id,
            "response": result.content,
        }
    finally:
        close_trace_context(handle)


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()
    enable_mlflow_langchain_enrichment()

    chain = build_chain()
    session_ids = [
        "session-1",
        "session-1",
        "session-1",
        "session-2",
        "session-2",
        "session-2",
        "session-3",
        "session-3",
        "session-3",
        "session-3",
    ]

    tasks = [
        run_request(
            chain,
            request_id=index + 1,
            session_id=session_id,
            experiment_id=experiment.experiment_id,
        )
        for index, session_id in enumerate(session_ids)
    ]
    results = await asyncio.gather(*tasks)

    print("\nSummary:")
    for item in results:
        print(
            f"request={item['request_id']} session={item['session_id']} "
            f"response={item['response'][:80]}"
        )


if __name__ == "__main__":
    asyncio.run(main())

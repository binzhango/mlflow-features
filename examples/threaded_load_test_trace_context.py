"""Verify trace-context isolation with concurrent threaded LangChain requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
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
        {"run_name": "support-assistant-threaded-load"}
    )


def run_request(chain, request_id: int, session_id: str) -> dict[str, str]:
    handle = open_trace_context(
        user_id=f"user-{request_id:02d}",
        session_id=session_id,
        client_request_id=f"req-20260306-thread-{request_id:02d}",
        mlflow_run_name=f"thread-load-{session_id}-{request_id:02d}",
        ensure_run=True,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "threaded-load-test",
            "session_group": session_id,
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "request_number": str(request_id),
        },
        run_tags={"team": "support", "scenario": "threaded-load-test"},
        span_metadata={"tenant": "local-demo", "provider": "ollama"},
        trace_name=f"thread-load-{session_id}",
    )

    try:
        current = get_current_trace_context()
        print(
            f"start request={request_id:02d} session={session_id} "
            f"context_session={current.session_id}"
        )

        result = chain.invoke(
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


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
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

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(run_request, chain, index + 1, session_id)
            for index, session_id in enumerate(session_ids)
        ]
        results = [future.result() for future in as_completed(futures)]

    print("\nSummary:")
    for item in sorted(results, key=lambda row: row["request_id"]):
        print(
            f"request={item['request_id']} session={item['session_id']} "
            f"response={item['response'][:80]}"
        )


if __name__ == "__main__":
    main()

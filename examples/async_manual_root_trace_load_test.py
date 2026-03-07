"""Verify concurrent async requests using a manual MLflow root trace per request."""

from __future__ import annotations

import asyncio

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import using_root_trace


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise support assistant."),
            ("human", "{question}"),
        ]
    )
    return (prompt | ChatOllama(model="nemotron-3-nano", temperature=0)).with_config(
        {"run_name": "support-assistant-async-manual-root"}
    )

async def traced_request(
    chain,
    *,
    request_id: int,
    session_id: str,
) -> dict[str, str]:
    with using_root_trace(
        user_id=f"user-{request_id:02d}",
        session_id=session_id,
        client_request_id=f"req-20260306-manual-root-{request_id:02d}",
        mlflow_run_name=f"async-manual-root-{session_id}-{request_id:02d}",
        ensure_run=True,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "async-manual-root-trace",
            "session_group": session_id,
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "request_number": str(request_id),
        },
        request_preview=(
            f"Request {request_id:02d} for {session_id}: summarize potential double-charge cause."
        ),
        trace_name=f"async-manual-root-{session_id}",
    ) as trace:
        payload = {
            "question": (
                f"Request {request_id:02d}: summarize why invoice INV-{request_id:02d} "
                f"might be double charged for session {session_id}."
            )
        }
        trace.request = payload
        result = await chain.ainvoke(payload)
        trace.response = result

    return {
        "request_id": f"{request_id:02d}",
        "session_id": session_id,
        "response": result.content,
    }


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    chain = build_chain()
    session_ids = [
        "session-11",
        "session-11",
        "session-11",
        "session-22",
        "session-22",
        "session-2",
        "session-33",
        "session-33",
        "session-33",
        "session-33",
    ]

    tasks = [
        traced_request(
            chain,
            request_id=index + 1,
            session_id=session_id,
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

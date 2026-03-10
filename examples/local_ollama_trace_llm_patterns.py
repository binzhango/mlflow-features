"""Local MLflow example showing three ergonomic tracing patterns."""

from __future__ import annotations

import asyncio

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import (
    auto_trace_llm,
    trace_llm,
    trace_llm_call,
)


def build_messages(question: str):
    return [
        SystemMessage(content="You are a concise support assistant."),
        HumanMessage(content=question),
    ]


@trace_llm_call(
    user_id="user-103",
    session_id="session-20260310-decorator-001",
    trace_name="decorator-pattern",
    mlflow_run_name="decorator-pattern-demo",
    ensure_run=True,
    tags={"app": "support-bot", "feature": "decorator-pattern"},
    metadata={"app_version": "0.1.0", "deployment": "local-mlflow-server"},
)
async def traced_agent(llm, input):
    return await llm.ainvoke(input)


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    llm = ChatOllama(model="nemotron-3-nano", temperature=0)

    auto_llm = auto_trace_llm(
        llm,
        user_id="user-101",
        session_id="session-20260310-auto-001",
        trace_name="auto-pattern",
        mlflow_run_name="auto-pattern-demo",
        ensure_run=True,
        tags={"app": "support-bot", "feature": "auto-pattern", "model": llm.model},
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "model_name": llm.model,
        },
        tags_builder=lambda runnable, inputs, response, error: {
            "response_kind": "tool" if getattr(response, "tool_calls", None) else "chat",
        },
        metadata_builder=lambda runnable, inputs, response, error: {
            "response_chars": len(getattr(response, "content", "") or ""),
        },
    )
    auto_response = await auto_llm.ainvoke(
        build_messages("Why was invoice INV-42 charged twice?")
    )
    print("1. auto_trace_llm(...)")
    print(auto_response.content)

    print("\n2. async with trace_llm(...)")
    context_messages = build_messages("Summarize the likely cause in one sentence.")
    async with trace_llm(
        user_id="user-102",
        session_id="session-20260310-context-001",
        trace_name="context-pattern",
        mlflow_run_name="context-pattern-demo",
        ensure_run=True,
        tags={"app": "support-bot", "feature": "context-pattern", "model": llm.model},
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "model_name": llm.model,
        },
    ) as trace:
        trace.set_request(context_messages)
        context_response = await llm.ainvoke(context_messages)
        trace.add_tags({"model": llm.model})
        trace.add_metadata(
            {
                "model_name": llm.model,
                "response_chars": len(getattr(context_response, "content", "") or ""),
            }
        )
        trace.set_response(context_response)
        print(context_response.content)

    print("\n3. @trace_llm_call(...)")
    decorated_response = await traced_agent(
        llm,
        build_messages("Reply with a short billing triage summary."),
    )
    print(decorated_response.content)


if __name__ == "__main__":
    asyncio.run(main())

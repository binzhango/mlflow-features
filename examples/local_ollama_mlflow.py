"""Minimal local example for MLflow 3 + Ollama + LangChain autolog enrichment."""

from __future__ import annotations

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import TraceContext, invoke_with_enrichment, using_trace_context


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise support assistant."),
            ("human", "{question}"),
        ]
    )
    llm = ChatOllama(model="nemotron-3-nano", temperature=0)

    chain = (prompt | llm).with_config(
        {"run_name": "support-assistant"}
    )

    trace_context = TraceContext(
        user_id="user-42",
        session_id="session-20260306-001",
        client_request_id="req-20260306-abc",
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "billing-help",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
        },
        span_metadata={
            "tenant": "local-demo",
            "provider": "ollama",
            "entrypoint": "examples/local_ollama_mlflow.py",
        },
        trace_name="billing-chat",
    )

    result = invoke_with_enrichment(
        chain,
        {"question": "Why was invoice INV-42 charged twice?"},
        trace_context,
        config={"metadata": {"route": "billing"}},
    )

    print(result.content)


    with using_trace_context(
        user_id="user-42",
        session_id="session-20260306-001",
        client_request_id="req-20260306-abc",
        mlflow_run_name="support-auto-enrichment-run",
        ensure_run=True,
        # tags={
        #     "app": "support-bot",
        #     "environment": "dev",
        #     "feature": "auto-enrichment",
        # },
        # metadata={
        #     "app_version": "0.1.0",
        #     "deployment": "local-mlflow-server",
        # },
        # run_tags={"team": "support"},
        span_metadata={"tenant": "local-demo", "provider": "ollama"},
        trace_name="support-auto-enrichment",
    ):
        result = llm.invoke({"question": "Why was invoice INV-42 charged twice?"})

    print(result.content)


if __name__ == "__main__":
    main()

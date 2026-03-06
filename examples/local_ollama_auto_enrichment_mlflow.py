"""Local MLflow 3 example with auto-enrichment and unchanged invoke() usage."""

from __future__ import annotations

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import enable_mlflow_langchain_enrichment, using_trace_context


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    # One-time startup hook. Existing chain.invoke(...) calls can remain unchanged.
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

    with using_trace_context(
        user_id="user-42",
        session_id="session-20260306-auto-001",
        client_request_id="req-20260306-auto-abc",
        mlflow_run_name="support-auto-enrichment-run",
        ensure_run=True,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "auto-enrichment",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
        },
        run_tags={"team": "support"},
        span_metadata={"tenant": "local-demo", "provider": "ollama"},
        trace_name="support-auto-enrichment",
    ):
        result = chain.invoke({"question": "Why was invoice INV-42 charged twice?"})

    print(result.content)


if __name__ == "__main__":
    main()

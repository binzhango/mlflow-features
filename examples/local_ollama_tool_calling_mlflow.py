"""Local MLflow 3 example for LangChain 1.x tool-calling traces with Ollama."""

from __future__ import annotations

import mlflow
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import TraceContext, invoke_with_enrichment


@tool
def lookup_account_tier(account_id: str) -> str:
    """Look up the support tier for an account."""
    tiers = {
        "ACME-42": "enterprise",
        "STARTUP-7": "pro",
        "DEFAULT-1": "free",
    }
    return tiers.get(account_id, "standard")


@tool
def lookup_ticket_status(ticket_id: str) -> str:
    """Look up the current status of a support ticket."""
    statuses = {
        "TICK-1001": "Open and assigned to the billing escalation team.",
        "TICK-1002": "Resolved after duplicate-charge refund was issued.",
    }
    return statuses.get(ticket_id, "Ticket not found.")


def build_agent():
    return create_agent(
        model=ChatOllama(model="nemotron-3-nano", temperature=0),
        tools=[lookup_account_tier, lookup_ticket_status],
        system_prompt=(
            "You are a support assistant. Use tools for account and ticket questions. "
            "Answer concisely and mention which tool-backed facts you used."
        ),
        debug=True,
        name="support-tool-agent",
    )


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    agent = build_agent()

    trace_context = TraceContext(
        user_id="user-42",
        session_id="session-20260306-tool-001",
        client_request_id="req-20260306-tools-abc",
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "tool-calling",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
        },
        span_metadata={
            "tenant": "local-demo",
            "provider": "ollama",
            "entrypoint": "examples/local_ollama_tool_calling_mlflow.py",
        },
        trace_name="support-tool-calling",
    )

    result = invoke_with_enrichment(
        agent,
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What tier is account ACME-42 on, and what is the status of ticket "
                        "TICK-1001? Use the available tools before answering."
                    ),
                }
            ]
        },
        trace_context,
        config={"metadata": {"route": "support-tool-calling"}},
    )

    print("Messages:")
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            print(f"- tool[{message.name}] -> {message.content}")
        elif isinstance(message, AIMessage):
            print(f"- ai -> {message.content}")
        else:
            print(f"- {type(message).__name__} -> {getattr(message, 'content', message)}")


if __name__ == "__main__":
    main()

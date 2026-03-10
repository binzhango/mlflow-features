"""Supervisor agent example with three traced subagents sharing one session id."""

from __future__ import annotations

import asyncio

import mlflow
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import using_root_trace


def build_specialist_agent(
    *,
    name: str,
    system_prompt: str,
):
    return create_agent(
        model=ChatOllama(model="nemotron-3-nano", temperature=0),
        tools=[],
        system_prompt=system_prompt,
        debug=True,
        name=name,
    )


def extract_final_ai_text(result: dict) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return ""


async def invoke_agent_with_trace(
    agent,
    *,
    agent_name: str,
    agent_role: str,
    question: str,
    session_id: str,
    user_id: str,
    request_suffix: str,
) -> dict:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    }

    with mlflow.start_run(
        run_name=f"{agent_name}-run",
        nested=mlflow.active_run() is not None,
    ):
        with using_root_trace(
            user_id=user_id,
            session_id=session_id,
            client_request_id=f"req-{session_id}-{request_suffix}",
            ensure_run=False,
            trace_name=agent_name,
            tags={
                "app": "support-bot",
                "environment": "dev",
                "feature": "supervisor-subagents",
                "agent_name": agent_name,
                "agent_role": agent_role,
            },
            metadata={
                "app_version": "0.1.0",
                "deployment": "local-mlflow-server",
                "entrypoint": "examples/local_ollama_supervisor_subagents_mlflow.py",
            },
        ) as trace:
            trace.request = payload
            result = await agent.ainvoke(
                payload,
                config={
                    "run_name": agent_name,
                    "metadata": {
                        "agent_name": agent_name,
                        "agent_role": agent_role,
                    },
                },
            )
            trace.response = result
            return result


def build_supervisor_agent(*, session_id: str, user_id: str):
    billing_agent = build_specialist_agent(
        name="billing-subagent",
        system_prompt=(
            "You are the billing specialist. Analyze duplicate-charge questions and respond "
            "with a concise billing-focused explanation."
        ),
    )
    account_agent = build_specialist_agent(
        name="account-subagent",
        system_prompt=(
            "You are the account specialist. Summarize account context such as support tier "
            "and customer priority in one concise paragraph."
        ),
    )
    escalation_agent = build_specialist_agent(
        name="escalation-subagent",
        system_prompt=(
            "You are the escalation specialist. Decide whether the case should be escalated "
            "and provide a short recommendation."
        ),
    )

    @tool
    async def ask_billing_specialist(question: str) -> str:
        """Ask the billing subagent to analyze the billing issue."""

        result = await invoke_agent_with_trace(
            billing_agent,
            agent_name="billing-subagent",
            agent_role="specialist",
            question=question,
            session_id=session_id,
            user_id=user_id,
            request_suffix="billing",
        )
        return extract_final_ai_text(result)

    @tool
    async def ask_account_specialist(question: str) -> str:
        """Ask the account subagent to summarize account context."""

        result = await invoke_agent_with_trace(
            account_agent,
            agent_name="account-subagent",
            agent_role="specialist",
            question=question,
            session_id=session_id,
            user_id=user_id,
            request_suffix="account",
        )
        return extract_final_ai_text(result)

    @tool
    async def ask_escalation_specialist(question: str) -> str:
        """Ask the escalation subagent whether this case should be escalated."""

        result = await invoke_agent_with_trace(
            escalation_agent,
            agent_name="escalation-subagent",
            agent_role="specialist",
            question=question,
            session_id=session_id,
            user_id=user_id,
            request_suffix="escalation",
        )
        return extract_final_ai_text(result)

    supervisor_agent = create_agent(
        model=ChatOllama(model="nemotron-3-nano", temperature=0),
        tools=[
            ask_billing_specialist,
            ask_account_specialist,
            ask_escalation_specialist,
        ],
        system_prompt=(
            "You are the supervisor agent. For every request, you must call all three specialist "
            "tools exactly once: ask_account_specialist, ask_billing_specialist, and "
            "ask_escalation_specialist. After all tool calls return, produce a concise final "
            "answer that includes account context, billing cause, and escalation recommendation."
        ),
        debug=True,
        name="supervisor-agent",
    )
    return supervisor_agent


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    session_id = "session-20260310-supervisor-001"
    user_id = "user-42"
    supervisor_agent = build_supervisor_agent(session_id=session_id, user_id=user_id)

    question = (
        "Customer ACME-42 says invoice INV-42 was charged twice. Gather account context, "
        "explain the likely billing cause, and say whether this should be escalated. "
        "Use every specialist before the final answer."
    )

    result = await invoke_agent_with_trace(
        supervisor_agent,
        agent_name="supervisor-agent",
        agent_role="supervisor",
        question=question,
        session_id=session_id,
        user_id=user_id,
        request_suffix="supervisor",
    )

    print(f"Shared session_id: {session_id}")
    print("Expected run names:")
    print("- supervisor-agent-run")
    print("- billing-subagent-run")
    print("- account-subagent-run")
    print("- escalation-subagent-run")

    print("\nMessages:")
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            print(f"- tool[{message.name}] -> {message.content}")
        elif isinstance(message, AIMessage):
            print(f"- ai -> {message.content}")
        else:
            print(f"- {type(message).__name__} -> {getattr(message, 'content', message)}")


if __name__ == "__main__":
    asyncio.run(main())

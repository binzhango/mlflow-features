"""Deep Agents supervisor example using auto_trace_llm() for supervisor and subagents."""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent
import mlflow
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import auto_trace_llm

def build_traced_llm(
    *,
    agent_name: str,
    agent_role: str,
    session_id: str,
    user_id: str,
) -> ChatOllama:
    base_llm = ChatOllama(model="nemotron-3-nano", temperature=0).with_config(
        {"run_name": agent_name}
    )
    return auto_trace_llm(
        base_llm,
        user_id=user_id,
        session_id=session_id,
        client_request_id=f"req-{session_id}-{agent_name}",
        ensure_run=False,
        trace_name=agent_name,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "deepagents-supervisor",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "model": getattr(base_llm, "model", "unknown"),
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_deepagents_supervisor_mlflow.py",
            "model_name": getattr(base_llm, "model", "unknown"),
        },
    )


def extract_final_ai_text(result: dict) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return ""


def build_compiled_subagent(
    *,
    agent_name: str,
    description: str,
    system_prompt: str,
    session_id: str,
    user_id: str,
) -> dict:
    llm = build_traced_llm(
        agent_name=agent_name,
        agent_role="specialist",
        session_id=session_id,
        user_id=user_id,
    )
    runnable = create_agent(
        model=llm,
        tools=[],
        system_prompt=(
            system_prompt
            + "\n\nReturn one concise final answer. Do not call tools unless absolutely required."
        ),
        name=agent_name,
    )
    return {
        "name": agent_name,
        "description": description,
        "runnable": runnable,
    }


def build_supervisor_agent(*, session_id: str, user_id: str):
    return create_deep_agent(
        model=build_traced_llm(
            agent_name="supervisor-agent",
            agent_role="supervisor",
            session_id=session_id,
            user_id=user_id,
        ),
        tools=[],
        system_prompt=(
            "You are the supervisor agent. Use the task tool exactly three times, once for each "
            "specialist: billing-subagent, account-subagent, and escalation-subagent. Ask each "
            "specialist for one concise final answer, then synthesize a short final response."
        ),
        subagents=[
            build_compiled_subagent(
                agent_name="billing-subagent",
                description="Use this subagent to analyze duplicate-charge and billing issues.",
                system_prompt=(
                    "You are the billing specialist. Analyze duplicate-charge questions and "
                    "respond with a concise billing-focused explanation."
                ),
                session_id=session_id,
                user_id=user_id,
            ),
            build_compiled_subagent(
                agent_name="account-subagent",
                description="Use this subagent to summarize account context and customer priority.",
                system_prompt=(
                    "You are the account specialist. Summarize account context such as support "
                    "tier and customer priority in one concise paragraph."
                ),
                session_id=session_id,
                user_id=user_id,
            ),
            build_compiled_subagent(
                agent_name="escalation-subagent",
                description="Use this subagent to decide whether the case should be escalated.",
                system_prompt=(
                    "You are the escalation specialist. Decide whether the case should be "
                    "escalated and provide a short recommendation."
                ),
                session_id=session_id,
                user_id=user_id,
            ),
        ],
        debug=True,
        name="supervisor-agent",
    )


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")

    session_id = "session-20260310-deepagents-003"
    user_id = "user-42"
    supervisor_agent = build_supervisor_agent(session_id=session_id, user_id=user_id)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Customer ACME-42 says invoice INV-42 was charged twice. Gather account "
                    "context, explain the likely billing cause, and say whether this should be "
                    "escalated. Use every subagent before the final answer."
                ),
            }
        ]
    }

    result = await supervisor_agent.ainvoke(payload)

    print(f"Shared session_id: {session_id}")
    print("Expected trace names / agent tags:")
    print("- supervisor-agent")
    print("- billing-subagent")
    print("- account-subagent")
    print("- escalation-subagent")
    print("\nFinal answer:")
    print(extract_final_ai_text(result))

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

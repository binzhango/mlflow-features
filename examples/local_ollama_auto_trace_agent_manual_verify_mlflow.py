"""Deterministic local verification example for auto_trace_agent().

This avoids model-driven delegation loops by orchestrating traced subagents directly in Python
while still using real LangChain agents for the subagents themselves.

Expected MLflow trace shape for one request:

- supervisor-agent
  - billing-subagent
    - billing-model
  - account-subagent
    - account-model
  - escalation-subagent
    - escalation-model
  - supervisor-synthesis-model
"""

from __future__ import annotations

import asyncio

import mlflow
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import auto_trace_agent, auto_trace_llm, using_root_trace


MODEL_NAME = "nemotron-3-nano"


def build_traced_model(
    *,
    span_name: str,
    session_id: str,
    user_id: str,
    agent_name: str,
    agent_role: str,
) -> ChatOllama:
    base_model = ChatOllama(model=MODEL_NAME, temperature=0).with_config({"run_name": span_name})
    return auto_trace_llm(
        base_model,
        session_id=session_id,
        user_id=user_id,
        trace_name=span_name,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "auto-trace-agent-manual-verify",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model": getattr(base_model, "model", MODEL_NAME),
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_manual_verify_mlflow.py",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model_name": getattr(base_model, "model", MODEL_NAME),
        },
    )


def build_specialist_agent(
    *,
    name: str,
    system_prompt: str,
    session_id: str,
    user_id: str,
):
    agent = create_agent(
        model=build_traced_model(
            span_name=f"{name}-model",
            session_id=session_id,
            user_id=user_id,
            agent_name=name,
            agent_role="specialist",
        ),
        tools=[],
        system_prompt=system_prompt,
        debug=True,
        name=name,
    )
    return auto_trace_agent(
        agent,
        session_id=session_id,
        user_id=user_id,
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "auto-trace-agent-manual-verify",
            "agent_name": name,
            "agent_role": "specialist",
            "span_kind": "agent",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_manual_verify_mlflow.py",
            "agent_name": name,
            "agent_role": "specialist",
            "span_kind": "agent",
        },
        model_max_retries=3,
    )


def extract_final_ai_text(result: dict) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return ""


async def ask_specialist(agent, question: str) -> str:
    payload = {"messages": [{"role": "user", "content": question}]}
    result = await agent.ainvoke(payload)
    return extract_final_ai_text(result)


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    session_id = "session-20260313-auto-trace-agent-manual-001"
    user_id = "user-42"

    billing_agent = build_specialist_agent(
        name="billing-subagent",
        system_prompt=(
            "You are the billing specialist. Explain the likely cause of the duplicate charge "
            "in one concise paragraph."
        ),
        session_id=session_id,
        user_id=user_id,
    )
    account_agent = build_specialist_agent(
        name="account-subagent",
        system_prompt=(
            "You are the account specialist. Summarize the customer's account context and "
            "priority in one concise paragraph."
        ),
        session_id=session_id,
        user_id=user_id,
    )
    escalation_agent = build_specialist_agent(
        name="escalation-subagent",
        system_prompt=(
            "You are the escalation specialist. Decide whether the case should be escalated "
            "and provide a short recommendation."
        ),
        session_id=session_id,
        user_id=user_id,
    )
    supervisor_synthesis_model = build_traced_model(
        span_name="supervisor-synthesis-model",
        session_id=session_id,
        user_id=user_id,
        agent_name="supervisor-agent",
        agent_role="supervisor",
    )

    user_question = (
        "Customer ACME-42 says invoice INV-42 was charged twice. Gather account context, "
        "explain the likely billing cause, and say whether this should be escalated."
    )
    root_request = {"messages": [{"role": "user", "content": user_question}]}

    with using_root_trace(
        session_id=session_id,
        user_id=user_id,
        trace_name="supervisor-agent",
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "auto-trace-agent-manual-verify",
            "agent_name": "supervisor-agent",
            "agent_role": "supervisor",
            "span_kind": "agent",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_manual_verify_mlflow.py",
            "agent_name": "supervisor-agent",
            "agent_role": "supervisor",
            "span_kind": "agent",
        },
    ) as trace:
        trace.request = root_request

        billing_result, account_result, escalation_result = await asyncio.gather(
            ask_specialist(
                billing_agent,
                "Analyze the likely billing cause for a duplicate invoice charge.",
            ),
            ask_specialist(
                account_agent,
                "Summarize account context, support tier, and customer priority.",
            ),
            ask_specialist(
                escalation_agent,
                "Decide whether this duplicate-charge case should be escalated.",
            ),
        )

        synthesis_prompt = [
            {
                "role": "system",
                "content": (
                    "You are the supervisor agent. Combine the specialist outputs into one short "
                    "final answer for the support operator."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original request: {user_question}\n\n"
                    f"Billing specialist: {billing_result}\n\n"
                    f"Account specialist: {account_result}\n\n"
                    f"Escalation specialist: {escalation_result}\n\n"
                    "Return one concise final answer."
                ),
            },
        ]
        final_response = await supervisor_synthesis_model.ainvoke(synthesis_prompt)
        final_text = getattr(final_response, "content", str(final_response))
        trace.response = {"answer": final_text}

    print(f"session_id: {session_id}")
    print("expected root span: supervisor-agent")
    print("expected child agent spans:")
    print("- billing-subagent")
    print("- account-subagent")
    print("- escalation-subagent")
    print("expected nested model spans:")
    print("- billing-model")
    print("- account-model")
    print("- escalation-model")
    print("- supervisor-synthesis-model")
    print("\nOpen the MLflow trace UI and confirm all spans share one trace_id.")
    print("Each subagent span should be a child of supervisor-agent.")
    print("\nFinal answer:")
    print(final_text)


if __name__ == "__main__":
    asyncio.run(main())

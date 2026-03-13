"""Deterministic-ish Deep Agents example with simple tools and agent tracing.

This example is designed to verify the full tracing shape with less model drift:

- one traced supervisor agent span
- one shared trace_id across the whole request
- one child span per subagent task invocation
- nested model spans inside each agent/subagent span
- simple tool usage inside each subagent

Expected agent/subagent span names:

- support-supervisor
- account-subagent
- billing-subagent
- escalation-subagent
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent
import mlflow
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import auto_trace_agent, auto_trace_llm


REQUIRED_SUBAGENTS = (
    "account-subagent",
    "billing-subagent",
    "escalation-subagent",
)

ACCOUNT_FIXTURE = {
    "account_id": "ACME-42",
    "support_tier": "enterprise",
    "priority": "high",
    "owner": "finance-ops",
}

INVOICE_FIXTURE = {
    "invoice_id": "INV-42",
    "status": "paid",
    "duplicate_charge_detected": True,
    "duplicate_charge_reason": "payment webhook replayed after a timeout",
}

ESCALATION_POLICY_FIXTURE = {
    "requires_escalation": True,
    "reason": "enterprise customer with confirmed duplicate charge",
    "target_team": "billing-escalations",
}


def _extract_task_history(messages: list[Any]) -> tuple[dict[str, str], list[str]]:
    tool_call_to_subagent: dict[str, str] = {}
    completed_results: dict[str, str] = {}
    ordered_attempts: list[str] = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict) or tool_call.get("name") != "task":
                    continue
                args = tool_call.get("args") or {}
                subagent_type = args.get("subagent_type") if isinstance(args, dict) else None
                tool_call_id = tool_call.get("id")
                if isinstance(subagent_type, str) and isinstance(tool_call_id, str):
                    tool_call_to_subagent[tool_call_id] = subagent_type
                    ordered_attempts.append(subagent_type)

        if isinstance(message, ToolMessage):
            subagent_type = tool_call_to_subagent.get(message.tool_call_id or "")
            if subagent_type is not None:
                completed_results[subagent_type] = str(message.content)

    return completed_results, ordered_attempts


def _next_missing_subagent(completed_results: dict[str, str]) -> str | None:
    for name in REQUIRED_SUBAGENTS:
        if name not in completed_results:
            return name
    return None


class EnforceOrderedSubagentsMiddleware(AgentMiddleware):
    """Keep the supervisor on a fixed subagent checklist to reduce model drift."""

    def _system_instruction(
        self,
        *,
        original: SystemMessage | None,
        next_subagent: str | None,
        finished: bool,
    ) -> SystemMessage:
        if finished:
            addition = (
                "All required subagents are complete. Do not call the task tool again. "
                "Write the final answer now using the existing tool results."
            )
        else:
            addition = (
                f"You must call the task tool next with subagent_type='{next_subagent}'. "
                "Do not skip ahead. Do not produce the final answer until all required "
                "subagents have completed in order."
            )

        if original is None:
            return SystemMessage(content=addition)
        return SystemMessage(content=f"{original.content}\n\n{addition}")

    def wrap_model_call(self, request, handler):
        completed_results, _ = _extract_task_history(request.state["messages"])
        next_subagent = _next_missing_subagent(completed_results)
        request = request.override(
            system_message=self._system_instruction(
                original=request.system_message,
                next_subagent=next_subagent,
                finished=next_subagent is None,
            )
        )
        return handler(request)

    async def awrap_model_call(self, request, handler):
        completed_results, _ = _extract_task_history(request.state["messages"])
        next_subagent = _next_missing_subagent(completed_results)
        request = request.override(
            system_message=self._system_instruction(
                original=request.system_message,
                next_subagent=next_subagent,
                finished=next_subagent is None,
            )
        )
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        if request.tool_call.get("name") != "task":
            return handler(request)

        args = request.tool_call.get("args") or {}
        subagent_type = args.get("subagent_type") if isinstance(args, dict) else None
        completed_results, _ = _extract_task_history(request.state["messages"])
        next_subagent = _next_missing_subagent(completed_results)

        if next_subagent is None:
            return ToolMessage(
                content="All required subagents already completed. Write the final answer now.",
                tool_call_id=request.tool_call["id"],
                name="task",
            )

        if subagent_type != next_subagent:
            return ToolMessage(
                content=(
                    f"Wrong subagent order. The next required subagent is {next_subagent}. "
                    "Call that subagent instead."
                ),
                tool_call_id=request.tool_call["id"],
                name="task",
            )

        return handler(request)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call.get("name") != "task":
            return await handler(request)

        args = request.tool_call.get("args") or {}
        subagent_type = args.get("subagent_type") if isinstance(args, dict) else None
        completed_results, _ = _extract_task_history(request.state["messages"])
        next_subagent = _next_missing_subagent(completed_results)

        if next_subagent is None:
            return ToolMessage(
                content="All required subagents already completed. Write the final answer now.",
                tool_call_id=request.tool_call["id"],
                name="task",
            )

        if subagent_type != next_subagent:
            return ToolMessage(
                content=(
                    f"Wrong subagent order. The next required subagent is {next_subagent}. "
                    "Call that subagent instead."
                ),
                tool_call_id=request.tool_call["id"],
                name="task",
            )

        return await handler(request)


def build_traced_model(
    *,
    model_name: str,
    session_id: str,
    user_id: str,
    agent_name: str,
    agent_role: str,
) -> ChatOllama:
    base_model = ChatOllama(model=model_name, temperature=0).with_config(
        {"run_name": f"{agent_name}-model"}
    )
    return auto_trace_llm(
        base_model,
        session_id=session_id,
        user_id=user_id,
        trace_name=f"{agent_name}-model",
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "auto-trace-agent-deepagents-tools",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model": getattr(base_model, "model", model_name),
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_deepagents_tools_mlflow.py",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model_name": getattr(base_model, "model", model_name),
        },
    )


@tool
def lookup_account_profile(account_id: str) -> str:
    """Return a tiny account profile for the given account id."""

    return (
        f"Account {account_id}: support_tier={ACCOUNT_FIXTURE['support_tier']}, "
        f"priority={ACCOUNT_FIXTURE['priority']}, owner={ACCOUNT_FIXTURE['owner']}"
    )


@tool
def lookup_invoice_status(invoice_id: str) -> str:
    """Return a tiny invoice summary for the given invoice id."""

    return (
        f"Invoice {invoice_id}: status={INVOICE_FIXTURE['status']}, "
        f"duplicate_charge_detected={INVOICE_FIXTURE['duplicate_charge_detected']}, "
        f"reason={INVOICE_FIXTURE['duplicate_charge_reason']}"
    )


@tool
def lookup_escalation_policy(issue_type: str) -> str:
    """Return the escalation policy for a given issue type."""

    return (
        f"Policy for {issue_type}: requires_escalation="
        f"{ESCALATION_POLICY_FIXTURE['requires_escalation']}, "
        f"reason={ESCALATION_POLICY_FIXTURE['reason']}, "
        f"target_team={ESCALATION_POLICY_FIXTURE['target_team']}"
    )


def build_deep_agent(*, session_id: str, user_id: str, model_name: str):
    deep_agent = create_deep_agent(
        model=build_traced_model(
            model_name=model_name,
            session_id=session_id,
            user_id=user_id,
            agent_name="support-supervisor",
            agent_role="supervisor",
        ),
        tools=[],
        middleware=[EnforceOrderedSubagentsMiddleware()],
        system_prompt=(
            "You are the support supervisor. Complete the investigation in this exact order: "
            "account-subagent, billing-subagent, escalation-subagent. Each subagent should "
            "return one concise result. After all three are done, write a short final answer "
            "with account context, billing cause, and escalation recommendation."
        ),
        subagents=[
            {
                "name": "account-subagent",
                "description": "Look up account context using a simple account tool.",
                "system_prompt": (
                    "You are the account specialist. Use the account lookup tool exactly once, "
                    "then return one concise paragraph with support tier, priority, and owner."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="account-subagent",
                    agent_role="specialist",
                ),
                "tools": [lookup_account_profile],
            },
            {
                "name": "billing-subagent",
                "description": "Inspect invoice status using a simple invoice tool.",
                "system_prompt": (
                    "You are the billing specialist. Use the invoice lookup tool exactly once, "
                    "then explain the likely billing cause in one concise paragraph."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="billing-subagent",
                    agent_role="specialist",
                ),
                "tools": [lookup_invoice_status],
            },
            {
                "name": "escalation-subagent",
                "description": "Check escalation policy using a simple policy tool.",
                "system_prompt": (
                    "You are the escalation specialist. Use the escalation policy tool exactly "
                    "once, then return a short escalation recommendation."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="escalation-subagent",
                    agent_role="specialist",
                ),
                "tools": [lookup_escalation_policy],
            },
        ],
        debug=True,
        name="support-supervisor",
    )

    return auto_trace_agent(
        deep_agent,
        session_id=session_id,
        user_id=user_id,
        trace_name="support-supervisor",
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "auto-trace-agent-deepagents-tools",
            "agent_name": "support-supervisor",
            "agent_role": "supervisor",
            "span_kind": "agent",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_deepagents_tools_mlflow.py",
            "agent_name": "support-supervisor",
            "agent_role": "supervisor",
            "span_kind": "agent",
        },
        model_max_retries=3,
    )


def extract_final_ai_text(result: dict[str, Any]) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return ""


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    # mlflow.langchain.autolog()

    session_id = "session-20260313-deepagents-tools-001"
    user_id = "user-43"
    model_name = "nemotron-3-nano"

    agent = build_deep_agent(
        session_id=session_id,
        user_id=user_id,
        model_name=model_name,
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Customer ACME-42 says invoice INV-42 was charged twice. "
                    "Use the subagents to gather account context, confirm the billing issue, "
                    "and decide whether the case should be escalated."
                ),
            }
        ]
    }

    result = await agent.ainvoke(
        payload,
        config={"recursion_limit": 60},
    )

    print(f"session_id: {session_id}")
    print("expected shared root span: support-supervisor")
    print("expected child subagent spans:")
    print("- account-subagent")
    print("- billing-subagent")
    print("- escalation-subagent")
    print("expected nested model spans:")
    print("- support-supervisor-model")
    print("- account-subagent-model")
    print("- billing-subagent-model")
    print("- escalation-subagent-model")
    print("expected simple tool usage inside subagents:")
    print("- lookup_account_profile")
    print("- lookup_invoice_status")
    print("- lookup_escalation_policy")
    print("\nOpen the MLflow trace UI and confirm all spans share one trace_id.")
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

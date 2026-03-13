"""Local Deep Agents verification example for auto_trace_agent().

Expected MLflow trace shape for one request:

- supervisor-agent
  - supervisor-model
  - billing-subagent
    - billing-model
  - account-subagent
    - account-model
  - escalation-subagent
    - escalation-model

If a subagent is called multiple times, you should see multiple child spans with the
same subagent name under the same supervisor trace.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent
import mlflow
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import auto_trace_agent, auto_trace_llm

REQUIRED_SUBAGENTS = (
    "billing-subagent",
    "account-subagent",
    "escalation-subagent",
)


def _extract_subagent_history(messages: list[Any]) -> tuple[dict[str, str], dict[str, int]]:
    tool_call_to_subagent: dict[str, str] = {}
    completed_results: dict[str, str] = {}
    attempted_counts: dict[str, int] = {}

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                if tool_call.get("name") != "task":
                    continue
                args = tool_call.get("args") or {}
                if not isinstance(args, dict):
                    continue
                subagent_type = args.get("subagent_type")
                tool_call_id = tool_call.get("id")
                if isinstance(subagent_type, str) and isinstance(tool_call_id, str):
                    tool_call_to_subagent[tool_call_id] = subagent_type
                    attempted_counts[subagent_type] = attempted_counts.get(subagent_type, 0) + 1

        if isinstance(message, ToolMessage):
            subagent_type = tool_call_to_subagent.get(message.tool_call_id or "")
            if subagent_type is not None:
                completed_results[subagent_type] = str(message.content)

    return completed_results, attempted_counts


class PreventDeepAgentLoopsMiddleware(AgentMiddleware):
    def _force_final_system_message(self, original: SystemMessage | None) -> SystemMessage:
        addition = (
            "All required subagents have already completed. Do not call the task tool again. "
            "Use the existing tool results and produce the final answer now."
        )
        if original is None:
            return SystemMessage(content=addition)
        return SystemMessage(content=f"{original.content}\n\n{addition}")

    def wrap_model_call(self, request, handler):
        completed_results, attempted_counts = _extract_subagent_history(request.state["messages"])
        if all(name in completed_results for name in REQUIRED_SUBAGENTS):
            request = request.override(
                system_message=self._force_final_system_message(request.system_message)
            )
        elif sum(attempted_counts.values()) >= 6:
            request = request.override(
                system_message=self._force_final_system_message(request.system_message)
            )
        return handler(request)

    async def awrap_model_call(self, request, handler):
        completed_results, attempted_counts = _extract_subagent_history(request.state["messages"])
        if all(name in completed_results for name in REQUIRED_SUBAGENTS):
            request = request.override(
                system_message=self._force_final_system_message(request.system_message)
            )
        elif sum(attempted_counts.values()) >= 6:
            request = request.override(
                system_message=self._force_final_system_message(request.system_message)
            )
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        if request.tool_call.get("name") != "task":
            return handler(request)

        args = request.tool_call.get("args") or {}
        subagent_type = args.get("subagent_type") if isinstance(args, dict) else None
        if not isinstance(subagent_type, str):
            return handler(request)

        completed_results, _ = _extract_subagent_history(request.state["messages"])
        if subagent_type in completed_results:
            return ToolMessage(
                content=(
                    f"Subagent {subagent_type} already completed earlier in this run. "
                    f"Reuse this result and move to the final answer:\n\n{completed_results[subagent_type]}"
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
        if not isinstance(subagent_type, str):
            return await handler(request)

        completed_results, _ = _extract_subagent_history(request.state["messages"])
        if subagent_type in completed_results:
            return ToolMessage(
                content=(
                    f"Subagent {subagent_type} already completed earlier in this run. "
                    f"Reuse this result and move to the final answer:\n\n{completed_results[subagent_type]}"
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
            "feature": "auto-trace-agent-deepagents",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model": getattr(base_model, "model", model_name),
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_deepagents_mlflow.py",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model_name": getattr(base_model, "model", model_name),
        },
    )


def build_deep_agent(*, session_id: str, user_id: str, model_name: str):
    deep_agent = create_deep_agent(
        model=build_traced_model(
            model_name=model_name,
            session_id=session_id,
            user_id=user_id,
            agent_name="supervisor-agent",
            agent_role="supervisor",
        ),
        tools=[],
        middleware=[PreventDeepAgentLoopsMiddleware()],
        system_prompt=(
            "You are the supervisor agent. You must call the task tool exactly three times, "
            "once each for billing-subagent, account-subagent, and escalation-subagent. "
            "After all three subagents return, provide a concise final answer. Do not call "
            "any subagent more than once. Do not call the task tool again after all three "
            "subagents have responded."
        ),
        subagents=[
            {
                "name": "billing-subagent",
                "description": "Analyze duplicate-charge and invoice issues.",
                "system_prompt": (
                    "You are the billing specialist. Explain the likely billing cause in one "
                    "concise paragraph."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="billing-subagent",
                    agent_role="specialist",
                ),
                "tools": [],
            },
            {
                "name": "account-subagent",
                "description": "Summarize account context and customer priority.",
                "system_prompt": (
                    "You are the account specialist. Summarize account context, support tier, "
                    "and customer priority in one concise paragraph."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="account-subagent",
                    agent_role="specialist",
                ),
                "tools": [],
            },
            {
                "name": "escalation-subagent",
                "description": "Decide whether the case should be escalated.",
                "system_prompt": (
                    "You are the escalation specialist. Decide whether the case should be "
                    "escalated and provide a short recommendation."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="escalation-subagent",
                    agent_role="specialist",
                ),
                "tools": [],
            },
        ],
        debug=True,
        name="supervisor-agent",
    )

    return auto_trace_agent(
        deep_agent,
        session_id=session_id,
        user_id=user_id,
        trace_name="supervisor-agent",
        tags={
            "app": "support-bot",
            "environment": "dev",
            "feature": "auto-trace-agent-deepagents",
            "agent_name": "supervisor-agent",
            "agent_role": "supervisor",
            "span_kind": "agent",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_deepagents_mlflow.py",
            "agent_name": "supervisor-agent",
            "agent_role": "supervisor",
            "span_kind": "agent",
        },
        model_max_retries=3,
    )


def extract_final_ai_text(result: dict) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return ""


async def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    # mlflow.langchain.autolog()

    session_id = "session-20260313-auto-trace-agent-003"
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
                    "Customer ACME-42 says invoice INV-42 was charged twice. Gather account "
                    "context, explain the likely billing cause, and say whether this should be "
                    "escalated. You must use every subagent before the final answer."
                ),
            }
        ]
    }

    result = await agent.ainvoke(
        payload,
        config={"recursion_limit": 100},
    )

    print(f"session_id: {session_id}")
    print("expected top-level trace/span name: supervisor-agent")
    print("expected child agent spans:")
    print("- billing-subagent")
    print("- account-subagent")
    print("- escalation-subagent")
    print("expected nested model spans:")
    print("- supervisor-model")
    print("- billing-model")
    print("- account-model")
    print("- escalation-model")
    print("invocation config: recursion_limit=100")
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

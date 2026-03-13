"""Deep Agents travel-planning example for agent-level MLflow tracing.

This example exercises:

- one traced supervisor agent span
- one shared trace across supervisor + subagents
- one child span per subagent invocation
- nested model spans for supervisor and subagents
- simple tool calls inside each subagent

Expected agent/subagent span names:

- trip-supervisor
- itinerary-subagent
- budget-subagent
- packing-subagent
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
    "itinerary-subagent",
    "budget-subagent",
    "packing-subagent",
)

ATTRACTION_FIXTURE = {
    "city": "Kyoto",
    "spots": [
        "Fushimi Inari early morning walk",
        "Kiyomizu-dera and Higashiyama district",
        "Arashiyama bamboo grove at sunset",
    ],
}

BUDGET_FIXTURE = {
    "lodging_per_night_usd": 145,
    "food_per_day_usd": 55,
    "local_transport_per_day_usd": 18,
    "attractions_total_usd": 40,
}

WEATHER_FIXTURE = {
    "city": "Kyoto",
    "season": "spring",
    "forecast": "cool mornings, mild afternoons, occasional light rain",
    "recommended_items": [
        "light rain jacket",
        "walking shoes",
        "compact umbrella",
        "layering shirt",
    ],
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


class EnforceTripSubagentOrderMiddleware(AgentMiddleware):
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
                "Write the final travel recommendation now."
            )
        else:
            addition = (
                f"You must call the task tool next with subagent_type='{next_subagent}'. "
                "Do not skip ahead. Do not finalize until all required subagents finish."
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
            "app": "travel-planner",
            "environment": "dev",
            "feature": "auto-trace-agent-deepagents-trip-tools",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model": getattr(base_model, "model", model_name),
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_deepagents_trip_tools_mlflow.py",
            "agent_name": agent_name,
            "agent_role": agent_role,
            "span_kind": "model",
            "model_name": getattr(base_model, "model", model_name),
        },
    )


@tool
def lookup_city_highlights(city: str) -> str:
    """Return a few sightseeing highlights for the requested city."""

    if city.lower() != ATTRACTION_FIXTURE["city"].lower():
        return f"No cached itinerary data for {city}."
    return "; ".join(ATTRACTION_FIXTURE["spots"])


@tool
def estimate_trip_cost(days: int) -> str:
    """Estimate a rough travel budget for the trip."""

    lodging_total = BUDGET_FIXTURE["lodging_per_night_usd"] * days
    food_total = BUDGET_FIXTURE["food_per_day_usd"] * days
    transport_total = BUDGET_FIXTURE["local_transport_per_day_usd"] * days
    attractions_total = BUDGET_FIXTURE["attractions_total_usd"]
    total = lodging_total + food_total + transport_total + attractions_total
    return (
        f"{days}-day estimate: lodging=${lodging_total}, food=${food_total}, "
        f"transport=${transport_total}, attractions=${attractions_total}, total=${total}"
    )


@tool
def lookup_weather_packing(city: str) -> str:
    """Return a simple weather summary and packing hints."""

    if city.lower() != WEATHER_FIXTURE["city"].lower():
        return f"No cached weather packing data for {city}."
    items = ", ".join(WEATHER_FIXTURE["recommended_items"])
    return (
        f"{WEATHER_FIXTURE['season']} in {city}: {WEATHER_FIXTURE['forecast']}. "
        f"Recommended items: {items}"
    )


def build_deep_agent(*, session_id: str, user_id: str, model_name: str):
    deep_agent = create_deep_agent(
        model=build_traced_model(
            model_name=model_name,
            session_id=session_id,
            user_id=user_id,
            agent_name="trip-supervisor",
            agent_role="supervisor",
        ),
        tools=[],
        middleware=[EnforceTripSubagentOrderMiddleware()],
        system_prompt=(
            "You are the trip supervisor. Plan a short Kyoto trip by completing the work in "
            "this exact order: itinerary-subagent, budget-subagent, packing-subagent. "
            "After all three return, write a concise final travel plan."
        ),
        subagents=[
            {
                "name": "itinerary-subagent",
                "description": "Build a small sightseeing plan using a cached highlights tool.",
                "system_prompt": (
                    "You are the itinerary specialist. Use the city highlights tool exactly "
                    "once, then return a two-day Kyoto outline in one concise paragraph."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="itinerary-subagent",
                    agent_role="specialist",
                ),
                "tools": [lookup_city_highlights],
            },
            {
                "name": "budget-subagent",
                "description": "Estimate the trip budget using a simple cost tool.",
                "system_prompt": (
                    "You are the budget specialist. Use the trip cost tool exactly once, then "
                    "return a short budget summary."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="budget-subagent",
                    agent_role="specialist",
                ),
                "tools": [estimate_trip_cost],
            },
            {
                "name": "packing-subagent",
                "description": "Suggest packing items using a cached weather tool.",
                "system_prompt": (
                    "You are the packing specialist. Use the weather packing tool exactly once, "
                    "then return a short packing recommendation."
                ),
                "model": build_traced_model(
                    model_name=model_name,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name="packing-subagent",
                    agent_role="specialist",
                ),
                "tools": [lookup_weather_packing],
            },
        ],
        debug=True,
        name="trip-supervisor",
    )

    return auto_trace_agent(
        deep_agent,
        session_id=session_id,
        user_id=user_id,
        trace_name="trip-supervisor",
        tags={
            "app": "travel-planner",
            "environment": "dev",
            "feature": "auto-trace-agent-deepagents-trip-tools",
            "agent_name": "trip-supervisor",
            "agent_role": "supervisor",
            "span_kind": "agent",
        },
        metadata={
            "app_version": "0.1.0",
            "deployment": "local-mlflow-server",
            "entrypoint": "examples/local_ollama_auto_trace_agent_deepagents_trip_tools_mlflow.py",
            "agent_name": "trip-supervisor",
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

    session_id = "session-20260313-trip-deepagents-001"
    user_id = "user-44"
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
                    "Plan a two-day Kyoto trip for a spring weekend. I want a small itinerary, "
                    "a rough budget, and a simple packing list."
                ),
            }
        ]
    }

    result = await agent.ainvoke(
        payload,
        config={"recursion_limit": 60},
    )

    print(f"session_id: {session_id}")
    print("expected shared root span: trip-supervisor")
    print("expected child subagent spans:")
    print("- itinerary-subagent")
    print("- budget-subagent")
    print("- packing-subagent")
    print("expected nested model spans:")
    print("- trip-supervisor-model")
    print("- itinerary-subagent-model")
    print("- budget-subagent-model")
    print("- packing-subagent-model")
    print("expected simple tool usage inside subagents:")
    print("- lookup_city_highlights")
    print("- estimate_trip_cost")
    print("- lookup_weather_packing")
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

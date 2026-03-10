# auto_trace_llm Guide

This guide explains how to use `auto_trace_llm(...)` as the default tracing API for plain LLM calls.

## What `auto_trace_llm(...)` Does

`auto_trace_llm(...)` wraps a LangChain LLM or chat model and opens/closes the MLflow root trace for each call.

It supports:

- `invoke(...)`
- `ainvoke(...)`
- `stream(...)`
- `astream(...)`

It keeps the original call shape unchanged:

```python
response = await llm.ainvoke(messages)
```

You do not need to create a special request object.

## Basic Usage

```python
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import auto_trace_llm


llm = ChatOllama(model="nemotron-3-nano", temperature=0)

llm = auto_trace_llm(
    llm,
    user_id="user-42",
    session_id="session-20260310-001",
    client_request_id="req-20260310-001",
    trace_name="support-chat",
    tags={
        "app": "support-bot",
        "environment": "dev",
    },
    metadata={
        "app_version": "0.1.0",
        "deployment": "local-mlflow-server",
    },
)

response = await llm.ainvoke(messages)
```

## What Gets Written

For each call, `auto_trace_llm(...)` writes:

- trace tags
- trace metadata
- `mlflow.trace.user`
- `mlflow.trace.session`
- request preview
- response preview
- root span `Inputs`
- root span `Outputs`

It also writes official MLflow token usage when the model response contains usage data:

- trace metadata: `mlflow.trace.tokenUsage`
- span attribute: `mlflow.chat.tokenUsage`

This is what populates the native `Tokens` column in the MLflow trace UI.

Readable copies are also written into metadata:

- `input_tokens`
- `output_tokens`
- `total_tokens`

## Static Tags and Metadata

Use wrap-time `tags` and `metadata` for fields that should apply to every call made through the wrapped LLM.

```python
llm = auto_trace_llm(
    llm,
    tags={
        "app": "support-bot",
        "environment": "prod",
    },
    metadata={
        "app_version": "1.2.3",
        "deployment": "us-east-1",
    },
)
```

Use this for:

- app name
- environment
- deployment
- version
- stable session/user identifiers

## Response-Based Enrichment

If the trace fields can be derived from `(runnable, inputs, response, error)`, define them up front with builders.

```python
def build_tags(runnable, inputs, response, error):
    return {
        "model": runnable.model,
    }


def build_metadata(runnable, inputs, response, error):
    return {
        "model_name": runnable.model,
        "response_type": type(response).__name__,
    }


llm = auto_trace_llm(
    llm,
    tags={"app": "support-bot"},
    metadata={"app_version": "0.1.0"},
    tags_builder=build_tags,
    metadata_builder=build_metadata,
)
```

Builder merge behavior:

1. base `tags` / `metadata`
2. builder output
3. builder values win on key collision

## Cost Calculation Example

If token usage is present on the response, you can calculate cost in `metadata_builder`.

```python
from decimal import Decimal


MODEL_PRICING = {
    "nemotron-3-nano": {
        "input_per_1k": Decimal("0.0000"),
        "output_per_1k": Decimal("0.0000"),
    }
}


def build_cost_metadata(runnable, inputs, response, error):
    if error is not None or response is None:
        return {}

    usage = getattr(response, "usage_metadata", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    model = getattr(runnable, "model", "unknown")
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return {"cost_usd": "unknown", "model_name": model}

    cost = (
        Decimal(input_tokens) / Decimal(1000) * pricing["input_per_1k"]
        + Decimal(output_tokens) / Decimal(1000) * pricing["output_per_1k"]
    )
    return {
        "model_name": model,
        "cost_usd": str(cost),
    }
```

## What You Cannot Do With `auto_trace_llm(...)`

You cannot imperatively do this after the wrapped call returns:

```python
response = await llm.ainvoke(messages)
# add tags here
# add metadata here
```

Reason:

- `auto_trace_llm(...)` closes the trace before control returns to caller code

If you need caller-side post-processing after the LLM returns, use `trace_llm(...)` instead.

## When To Use `trace_llm(...)` Instead

Use `trace_llm(...)` when:

- you need to inspect the response in caller code first
- you need `trace.add_tags(...)` or `trace.add_metadata(...)`
- you need custom logic that cannot be expressed as a wrap-time builder

Example:

```python
from mlflow_langchain_enrichment import trace_llm


async with trace_llm(
    user_id="user-42",
    session_id="session-20260310-001",
    tags={"app": "support-bot"},
) as trace:
    trace.set_request(messages)
    response = await llm.ainvoke(messages)
    trace.add_tags({"model": llm.model})
    trace.add_metadata({"model_name": llm.model})
    trace.set_response(response)
```

## Async and Load Tests

`auto_trace_llm(...)` works with concurrent async calls.

See:

- [examples/async_auto_trace_llm_load_test.py](/Users/binzhang/vibe_coding_repo/mlflow-features/examples/async_auto_trace_llm_load_test.py)
- [examples/async_auto_trace_llm_stream_test.py](/Users/binzhang/vibe_coding_repo/mlflow-features/examples/async_auto_trace_llm_stream_test.py)

## Streaming

`auto_trace_llm(...)` supports:

- `stream(...)`
- `astream(...)`

For streaming calls, the wrapper aggregates chunks and writes the final response when the stream completes.

## Deep Agents and Bound Chat Models

`auto_trace_llm(...)` now preserves chat-model compatibility for:

- direct chat models like `ChatOllama(...)`
- bound chat models like `ChatOllama(...).with_config({...})`

So this pattern is supported:

```python
llm = auto_trace_llm(
    ChatOllama(model="nemotron-3-nano", temperature=0).with_config(
        {"run_name": "supervisor-agent"}
    ),
    session_id="session-20260310-deepagents-001",
    trace_name="supervisor-agent",
)
```

This matters for frameworks like Deep Agents that expect a real `BaseChatModel`.

## Important Boundary: LLM Trace vs Agent Trace

`auto_trace_llm(...)` traces the LLM call boundary.

It does not guarantee one separate top-level trace per higher-level agent invocation.

So:

- plain LLM app: `auto_trace_llm(...)` is a good default
- agent orchestration: `auto_trace_llm(...)` is useful for model calls, but it is not the same as agent-level tracing

If you need one top-level trace per agent invocation, use an agent-level root trace wrapper instead.

## Important Async Caution

For async Deep Agents / LangGraph workflows, do not assume `mlflow.langchain.autolog()` and `auto_trace_llm(...)` should both be enabled together.

That combination can produce context warnings or unstable behavior in async task graphs.

Recommended rule:

- plain LangChain app: `mlflow.langchain.autolog()` may be fine
- async Deep Agents / LangGraph app: prefer `auto_trace_llm(...)` or explicit root tracing, not both

## Recommended Team Convention

For most teams, the simplest convention is:

1. wrap the shared LLM once with `auto_trace_llm(...)`
2. define common tags and metadata there
3. define reusable builder functions there for model name, token-based cost, or response type
4. keep business code unchanged as `await llm.ainvoke(input)`
5. use `trace_llm(...)` only when caller-side post-processing is required

## Related Guides

- [UI Attribute Guide](/Users/binzhang/vibe_coding_repo/mlflow-features/docs/ui-attributes-guide.md)
- [Performance Tuning Guide](/Users/binzhang/vibe_coding_repo/mlflow-features/docs/performance-tuning-guide.md)

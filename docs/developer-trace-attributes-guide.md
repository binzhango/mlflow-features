# Developer Trace Attribute Guide

This guide explains how developers should add:

- dynamic metadata
- new tags
- reserved MLflow metadata
- reserved MLflow tags

The goal is to keep trace enrichment consistent and avoid breaking MLflow UI behavior.

## Default Rule

Prefer the highest-level API that solves the problem.

Use this order:

1. `auto_trace_llm(...)` or `auto_trace_chain(...)` with static `tags` / `metadata`
2. `tags_builder` / `metadata_builder`
3. `trace_llm(...)` with `trace.add_tags(...)` / `trace.add_metadata(...)`
4. direct `mlflow.update_current_trace(...)` only for advanced cases

## Choose the Right Place

Use `tags` for:

- visible UI columns
- fields you want to scan quickly in the trace list

Examples:

- `app`
- `environment`
- `route`
- `model`

Use `metadata` for:

- searchable fields
- detail/debugging fields
- reserved trace metadata

Examples:

- `app_version`
- `deployment`
- `provider`
- `model_name`

## 1. Add Dynamic Metadata With Builders

If the value can be derived from `(runnable, inputs, response, error)`, use `metadata_builder`.

```python
from mlflow_langchain_enrichment import auto_trace_llm


def build_metadata(runnable, inputs, response, error):
    if error is not None or response is None:
        return {}
    return {
        "model_name": runnable.model,
        "response_type": type(response).__name__,
        "response_chars": len(getattr(response, "content", "") or ""),
    }


llm = auto_trace_llm(
    llm,
    metadata={"app_version": "0.1.0"},
    metadata_builder=build_metadata,
)
```

Merge rule:

1. base metadata
2. builder output
3. builder values win on key collision

## 2. Add New Tags With Builders

If the value should be a visible UI field, use `tags_builder`.

```python
def build_tags(runnable, inputs, response, error):
    if error is not None or response is None:
        return {}
    return {
        "model": runnable.model,
        "response_kind": "tool" if getattr(response, "tool_calls", None) else "chat",
    }


llm = auto_trace_llm(
    llm,
    tags={"app": "support-bot"},
    tags_builder=build_tags,
)
```

## 3. Add Attributes After the Call

If the value is computed in caller code after the response returns, use `trace_llm(...)`.

```python
from mlflow_langchain_enrichment import trace_llm


async with trace_llm(
    user_id="user-42",
    session_id="session-1",
    tags={"app": "support-bot"},
) as trace:
    trace.set_request(messages)
    response = await llm.ainvoke(messages)

    trace.add_tags({"model": llm.model})
    trace.add_metadata({"model_name": llm.model})
    trace.set_response(response)
```

Use this when:

- the value depends on caller-side branching
- the value is not convenient to express in a builder
- you want imperative control

## 4. Reserved Metadata: Preferred Paths

Prefer these high-level fields instead of writing raw reserved metadata when possible:

- `user_id` instead of `metadata["mlflow.trace.user"]`
- `session_id` instead of `metadata["mlflow.trace.session"]`
- `trace_name` instead of raw reserved trace-name tags

Example:

```python
llm = auto_trace_llm(
    llm,
    user_id="user-42",
    session_id="session-20260310-001",
    trace_name="support-chat",
)
```

This package will map:

- `user_id` -> `mlflow.trace.user`
- `session_id` -> `mlflow.trace.session`

## 5. Reserved Metadata You Can Write Manually

These are the main reserved MLflow trace metadata keys developers may need:

- `mlflow.trace.user`
- `mlflow.trace.session`
- `mlflow.trace.tokenUsage`
- `mlflow.modelId`

### Manual session/user example

If you are not using `user_id` / `session_id`, you can still write the reserved keys directly:

```python
llm = auto_trace_llm(
    llm,
    metadata={
        "mlflow.trace.user": "user-42",
        "mlflow.trace.session": "session-1",
    },
)
```

### Manual token usage example

The native MLflow `Tokens` UI column is driven by reserved token usage, not arbitrary metadata keys.

The value must be a JSON string like:

```python
{
    "input_tokens": 12,
    "output_tokens": 34,
    "total_tokens": 46,
}
```

Example:

```python
import json


def build_reserved_metadata(runnable, inputs, response, error):
    usage = {
        "input_tokens": 12,
        "output_tokens": 34,
        "total_tokens": 46,
    }
    return {
        "mlflow.trace.tokenUsage": json.dumps(usage, separators=(",", ":")),
    }
```

Note:

- this package now writes `mlflow.trace.tokenUsage` automatically when the wrapped response carries usage data
- for plain `auto_trace_llm(...)` or `auto_trace_chain(...)`, you usually do not need to set it yourself anymore

## 6. Reserved Span Attribute for Token Usage

MLflow also uses the reserved span attribute:

- `mlflow.chat.tokenUsage`

This package now writes that automatically for `auto_trace_llm(...)` and `auto_trace_chain(...)` when usage data is present.

You normally should not set it manually unless you are writing your own custom span logic.

## 7. Reserved Tags

Reserved trace tags exist in MLflow, for example:

- `mlflow.traceName`
- `mlflow.linkedPrompts`
- `mlflow.trace.spansLocation`

Developer rule:

- prefer high-level package arguments when available
- avoid writing reserved tags manually unless you understand the MLflow meaning of that tag

### Preferred way to set trace name

Use:

```python
llm = auto_trace_llm(
    llm,
    trace_name="support-chat",
)
```

Not:

```python
tags={"mlflow.traceName": "support-chat"}
```

### If you must write a reserved tag

Use the same places as normal tags:

- `tags=...`
- `tags_builder=...`
- `trace.add_tags(...)`

But do it intentionally and keep the key stable.

## 8. Direct `mlflow.update_current_trace(...)`

Use direct MLflow trace updates only when you really need low-level control.

Example:

```python
import mlflow

from mlflow_langchain_enrichment import trace_llm


async with trace_llm(session_id="session-1") as trace:
    trace.set_request(messages)
    response = await llm.ainvoke(messages)

    mlflow.update_current_trace(
        metadata={"custom_key": "custom_value"},
    )

    # Keep the local trace context aligned with the direct MLflow update.
    trace.add_metadata({"custom_key": "custom_value"})
    trace.set_response(response)
```

Important:

- if you call `mlflow.update_current_trace(...)` directly inside a `trace_llm(...)` block, also update the local `TraceSession`
- otherwise the final trace close may reapply stale local metadata/tags

## 9. Common Safe Patterns

### Static defaults for every call

```python
llm = auto_trace_llm(
    llm,
    tags={"app": "support-bot"},
    metadata={"app_version": "1.2.3"},
)
```

### Dynamic model metadata

```python
llm = auto_trace_llm(
    llm,
    metadata_builder=lambda runnable, inputs, response, error: {
        "model_name": runnable.model,
    },
)
```

### Dynamic token-based cost

```python
llm = auto_trace_llm(
    llm,
    metadata_builder=build_cost_metadata,
)
```

### Post-call enrichment

```python
async with trace_llm(session_id="session-1") as trace:
    trace.set_request(messages)
    response = await llm.ainvoke(messages)
    trace.add_metadata({"cost_bucket": "medium"})
    trace.set_response(response)
```

## 10. Avoid These Mistakes

- Do not expect arbitrary metadata like `total_tokens` to populate the MLflow `Tokens` column.
- Do not mix `mlflow.langchain.autolog()` with async Deep Agents or LangGraph flows unless you have verified the context behavior.
- Do not manually overwrite reserved MLflow keys casually. Some are internal and UI-sensitive.
- Do not use `auto_trace_llm(...)` when you need imperative caller-side post-processing after the call returns. Use `trace_llm(...)`.

## Recommended Team Convention

For most teams, use this standard:

1. add common static fields in `auto_trace_llm(...)` or `auto_trace_chain(...)`
2. add reusable response-derived fields in builders
3. use `trace_llm(...)` only when caller-side logic is necessary
4. use raw reserved metadata only for MLflow-specific features like token usage or model ID

## Related Guides

- [auto_trace_llm Guide](/Users/binzhang/vibe_coding_repo/mlflow-features/docs/auto-trace-llm-guide.md)
- [UI Attribute Guide](/Users/binzhang/vibe_coding_repo/mlflow-features/docs/ui-attributes-guide.md)
- [Performance Tuning Guide](/Users/binzhang/vibe_coding_repo/mlflow-features/docs/performance-tuning-guide.md)

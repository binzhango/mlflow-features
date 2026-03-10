# UI Attribute Guide

This guide explains how to add new attributes so they appear consistently in the MLflow trace UI.

The recommended APIs are:

- `auto_trace_llm(...)`
- `trace_llm(...)`
- `trace_llm_call(...)`

## Mental Model

There are three places to put information:

1. `tags`
   Use for values you want as visible trace-list columns.

2. `metadata`
   Use for searchable or diagnostic fields.

3. previews and span I/O
   Use for trace-list `Request` / `Response` text and root span `Inputs` / `Outputs`.

Practical rule:

- use `tags` for UI display
- use `metadata` for search and detail
- use previews when you want to control request/response text

## What This Package Writes

At trace close time, the package updates the current trace with:

- `tags`
- `metadata`
- `client_request_id`
- `request_preview`
- `response_preview`
- `state`

It also maps:

- `user_id` -> `metadata["mlflow.trace.user"]`
- `session_id` -> `metadata["mlflow.trace.session"]`

If root span I/O capture is enabled, it also writes:

- root span `Inputs`
- root span `Outputs`

## Which Field Should I Use?

### Visible UI column

Use `tags`.

Examples:

- `model`
- `app`
- `route`
- `tenant`
- `environment`

### Searchable detail field

Use `metadata`.

Examples:

- `app_version`
- `deployment`
- `model_name`
- `provider`

### Custom request/response text

Use:

- `request_preview`
- `response_preview`
- `request_preview_builder`
- `response_preview_builder`

## Two Supported Patterns

There are two main ways to add future UI attributes.

### 1. Configure them up front with `auto_trace_llm(...)`

Use this when the enrichment logic can be defined once and reused for every call.

Example:

```python
from mlflow_langchain_enrichment import auto_trace_llm


def build_tags(runnable, inputs, response, error):
    return {
        "model": runnable.model,
        "response_kind": "tool" if getattr(response, "tool_calls", None) else "chat",
    }


def build_metadata(runnable, inputs, response, error):
    return {
        "model_name": runnable.model,
        "response_chars": len(getattr(response, "content", "") or ""),
    }


llm = auto_trace_llm(
    llm,
    tags={"app": "support-bot"},
    metadata={"app_version": "0.1.0"},
    tags_builder=build_tags,
    metadata_builder=build_metadata,
)

response = await llm.ainvoke(messages)
```

This keeps the original `llm.ainvoke(messages)` signature unchanged.

Use this when:

- every call should use the same extraction logic
- the logic can be expressed as functions of `(runnable, inputs, response, error)`
- you do not need to imperatively mutate the trace in caller code

### 2. Update the open trace after the LLM call with `trace_llm(...)`

Use this when you need to inspect the response in caller code and then decide what to add.

Example:

```python
from mlflow_langchain_enrichment import trace_llm


def build_extra_tags(llm, inputs, response):
    return {"model": llm.model}


def build_extra_metadata(llm, inputs, response):
    return {
        "model_name": llm.model,
        "response_chars": len(getattr(response, "content", "") or ""),
    }


async with trace_llm(
    user_id=user_id,
    session_id=session_id,
    tags={"app": "support-bot"},
    metadata={"app_version": "0.1.0"},
) as trace:
    trace.set_request(messages)
    response = await llm.ainvoke(messages)
    trace.add_tags(build_extra_tags(llm, messages, response))
    trace.add_metadata(build_extra_metadata(llm, messages, response))
    trace.set_response(response)
```

This is the best choice when:

- the new attribute depends on caller-side logic after the model returns
- you want to call your own extractor function directly
- you want to keep `llm.ainvoke(input)` unchanged

## When To Use Which One

Use `auto_trace_llm(...)` when:

- the enrichment should be attached once at setup time
- teams should not need to think about trace updates inside business logic

Use `trace_llm(...)` when:

- the attribute is computed after the call
- the attribute depends on caller-side branching or post-processing
- you want explicit control over when the trace is updated

## Recommended Pattern For New Attributes

When adding a new UI field, decide:

1. Should it be visible in the trace list?
2. Is it only for search/debugging?
3. Is it known before the call, or only after the response?

Then apply these rules:

### Rule 1: Visible fields go in `tags`

Example:

```python
trace.add_tags({"model": llm.model})
```

### Rule 2: Detail or query fields go in `metadata`

Example:

```python
trace.add_metadata({"model_name": llm.model, "app_version": "0.1.0"})
```

### Rule 3: If it is useful in both places, store both

Example:

```python
trace.add_tags({"model": llm.model})
trace.add_metadata({"model_name": llm.model})
```

### Rule 4: Prefer stable field names

Good names:

- `model`
- `app`
- `route`
- `environment`
- `tenant`
- `provider`
- `app_version`

Avoid one-off names that will be hard to standardize later.

## Common Examples

### Model name

Visible UI column:

```python
trace.add_tags({"model": llm.model})
```

Search/detail field:

```python
trace.add_metadata({"model_name": llm.model})
```

### Response size

```python
trace.add_metadata(
    {"response_chars": len(getattr(response, "content", "") or "")}
)
```

### Tool-call marker

```python
trace.add_tags(
    {"has_tool_calls": "yes" if getattr(response, "tool_calls", None) else "no"}
)
```

### Error type

```python
trace.add_metadata({"error_type": type(error).__name__})
```

## Reusable Extractor Functions

Prefer named helpers over large inline lambdas.

Example:

```python
def build_model_tags(llm, inputs, response):
    return {"model": llm.model}


def build_model_metadata(llm, inputs, response):
    return {
        "model_name": llm.model,
        "response_chars": len(getattr(response, "content", "") or ""),
    }
```

Usage:

```python
response = await llm.ainvoke(messages)
trace.add_tags(build_model_tags(llm, messages, response))
trace.add_metadata(build_model_metadata(llm, messages, response))
```

## Team Standardization

If you want future UI additions to be consistent, define a small shared helper module such as:

```python
def build_standard_tags(llm, response):
    return {"model": llm.model}


def build_standard_metadata(llm, response):
    return {
        "model_name": llm.model,
        "app_version": "2026.03.10",
    }
```

Then reuse those functions everywhere instead of having each team invent field names independently.

## Run Name And Trace Name

These are different.

### `trace_name`

Controls the root trace/span name.

### `mlflow_run_name`

Controls the associated MLflow Run name if a run exists. If you want this reliably, also set:

```python
ensure_run=True
```

## Streaming Note

If you disable root span I/O capture:

```python
capture_root_span_io=False
```

then empty root span `Outputs` are expected.

If you want visible request-span outputs for streaming calls, keep root span I/O capture enabled.

## Troubleshooting

### I added metadata but do not see a visible column

That is usually expected. Use `tags` for visible trace-list columns.

### My run name is empty

Set:

- `mlflow_run_name`
- `ensure_run=True`

### My root span `Outputs` are empty

Check whether `capture_root_span_io=False`.

### I want to add fields after the model returns

Use `trace_llm(...)`, not just `auto_trace_llm(...)`.

## Practical Recommendation

For most teams:

1. use `auto_trace_llm(...)` for default tags and metadata
2. use `trace_llm(...)` when response-dependent enrichment is computed in caller code
3. standardize a small shared set of field names such as `model`, `app`, `route`, `environment`, `app_version`, and `model_name`

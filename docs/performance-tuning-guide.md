# Performance Tuning Guide

This guide explains how to reason about performance when using `mlflow_langchain_enrichment`, especially for async workloads.

It is written for future tuning work, not just current usage.

## Main Tradeoff

The core tradeoff is:

- lighter tracing is faster
- request-level root tracing is richer and more correct, but slower

If you need:

- request-level `Inputs` / `Outputs`
- stable per-request async isolation
- post-call tag/metadata updates

then you usually need the root-trace path:

- `trace_llm(...)`
- `auto_trace_llm(...)`
- `open_root_trace(...)`

If you only need lightweight enrichment and can accept weaker request-level trace envelopes, then `open_trace_context(...)` is usually cheaper.

## Where The Overhead Comes From

In the current implementation, the main overhead is not `session_id` or simple tags.

The more expensive operations are:

1. creating a root span for every request
2. updating the current trace on open and on close
3. generating request/response previews
4. optionally creating or attaching an MLflow Run
5. provider-side serialization or LangChain/MLflow async logging behavior

Code references:

- root span open: [auto.py](/Users/binzhang/vibe_coding_repo/mlflow-features/src/mlflow_langchain_enrichment/auto.py#L140)
- root span close: [auto.py](/Users/binzhang/vibe_coding_repo/mlflow-features/src/mlflow_langchain_enrichment/auto.py#L168)
- preview generation and trace update: [enrichment.py](/Users/binzhang/vibe_coding_repo/mlflow-features/src/mlflow_langchain_enrichment/enrichment.py#L237)
- auto-traced wrapper invoke path: [wrappers.py](/Users/binzhang/vibe_coding_repo/mlflow-features/src/mlflow_langchain_enrichment/wrappers.py#L139)

## Performance Profiles

### Fastest practical path

Use:

- `enable_mlflow_langchain_enrichment(...)`
- `open_trace_context(...)` or `using_trace_context(...)`

Characteristics:

- lower overhead
- no manual root request span
- less explicit request-level `Outputs`
- may be weaker for very concurrent async task isolation

### Correct async request envelope

Use:

- `trace_llm(...)`
- `auto_trace_llm(...)`
- `open_root_trace(...)`

Characteristics:

- correct request-level root span
- clear `Inputs` / `Outputs`
- supports post-call updates
- more overhead per request

## Current Tuning Knobs

### 1. `capture_root_span_io`

Controls whether root span `Inputs` / `Outputs` are written.

- `True`: write root span I/O
- `False`: keep the root span as an envelope

Important:

Turning this off does not remove most root-trace overhead. It only skips `set_inputs(...)` and `set_outputs(...)`.

It does not skip:

- opening the root span
- updating the current trace
- preview generation

Use it when:

- you want less duplicated data
- child model spans already contain enough I/O

Do not expect a large performance gain from this flag alone.

### 2. `ensure_run`

If `ensure_run=True`, the package may start an MLflow Run when no run is active.

This adds overhead.

If you do not need per-request run attachment, keep:

```python
ensure_run=False
```

### 3. Preview builders

Request and response preview generation can cost more than expected for large payloads.

Use these fields carefully:

- `request_preview`
- `response_preview`
- `request_preview_builder`
- `response_preview_builder`

Recommendations:

- use fixed strings when possible
- avoid expensive parsing
- avoid serializing large nested objects

### 4. Tag and metadata builders

Builders run on every request.

Recommendations:

- keep them cheap
- avoid heavy JSON parsing
- avoid network access
- avoid expensive token accounting unless necessary

Good:

```python
metadata_builder=lambda runnable, inputs, response, error: {
    "response_chars": len(getattr(response, "content", "") or ""),
}
```

Bad:

- scanning large prompt trees repeatedly
- expensive regex over long responses
- disk or network lookups

### 5. Destination routing

If all requests go to the same experiment, avoid changing destination inside every request unless needed.

Per-request destination setting is useful for async isolation, but it is still extra work.

If your destination is static, prefer setting it once at startup.

### 6. LLM object lifecycle

Do not confuse tracing overhead with model object construction overhead.

For fair measurements:

- reuse the same base LLM when possible
- compare tracing modes against the same model setup

Creating a new model client per task can dominate the benchmark and hide the actual tracing cost.

## Recommended Benchmark Strategy

When tuning performance, compare these paths with the same prompt, same model, same concurrency, and same experiment:

1. plain `llm.ainvoke(...)`
2. `open_trace_context(...)` plus unchanged `llm.ainvoke(...)`
3. `auto_trace_llm(...).ainvoke(...)`
4. `trace_llm(...)` plus manual `trace.set_request(...)` / `trace.set_response(...)`

Measure:

- total elapsed time
- p50 / p95 latency
- requests per second
- trace completeness in the UI

Without all four, it is hard to isolate where the overhead is coming from.

## What Usually Matters Most

In practice, the highest-cost parts are usually:

1. the model backend itself
2. MLflow/LangChain async logging behavior
3. root span lifecycle
4. custom preview or metadata builders

Simple static tags/metadata are usually cheap.

`session_id` and `user_id` by themselves should be negligible.

## Current Optimization Ideas

These are the most promising future improvements for this package.

### 1. Skip trace update on open

Current root tracing updates the current trace on open and again on close.

Potential optimization:

- add a flag like `update_trace_on_open=False`
- only update on close unless the user explicitly mutates the trace during execution

Why it matters:

- removes one MLflow trace update call per request
- likely higher impact than disabling root span I/O

### 2. Reduce repeated `TraceContext` allocation

The auto-traced wrapper currently rebuilds merged trace context state when applying dynamic attributes.

Potential optimization:

- mutate only the merged tags/metadata instead of reconstructing a fresh `TraceContext`

### 3. Add sampled root tracing

For high-volume traffic, sample only some requests into the full root-trace path.

Possible model:

- always attach lightweight trace context
- only open full root spans for sampled requests

### 4. Add a lightweight mode for auto tracing

Potential mode:

- no root span
- no root I/O
- final metadata/tags only

This would trade off UI richness for lower overhead.

### 5. Avoid unnecessary preview work

Potential optimization:

- allow preview generation to be disabled globally or per wrapper
- short-circuit preview generation when the user already provided explicit preview strings

## Async-Specific Notes

If async load testing still looks sequential, the bottleneck may not be this package.

Check:

- whether all tasks start quickly
- whether the provider serializes responses
- whether MLflow or LangChain async logging blocks the main thread

This package can add overhead, but provider-side serialization is often the dominant factor.

## Practical Guidance

Use this decision rule:

### Highest correctness

Use:

- `trace_llm(...)`
- `auto_trace_llm(...)`

Accept:

- more overhead

### Highest throughput

Use:

- `open_trace_context(...)`
- auto enrichment only

Accept:

- less explicit request-level `Outputs`
- weaker root request envelope

### Mixed strategy

Use:

- root tracing for critical requests
- lighter tracing for bulk traffic

This is often the best production compromise.

## Checklist For Future Tuning Work

Before changing code, check:

1. Are we benchmarking the same model setup across variants?
2. Is `ensure_run=True` really required?
3. Are preview builders doing expensive work?
4. Are we opening a root span when we only need metadata?
5. Are we updating the trace multiple times per request unnecessarily?
6. Is provider-side serialization dominating the results?

If the answer to 6 is yes, optimize the backend path before optimizing this package.

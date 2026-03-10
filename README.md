# mlflow-langchain-enrichment

Production-oriented helpers for enriching `mlflow.langchain.autolog()` traces with better trace list fields, request-scoped context, and async-friendly root tracing.

This package is meant for teams already using LangChain plus MLflow tracing who want the trace UI to show useful business context such as `session_id`, `user_id`, custom tags, and request/response previews without forcing every developer to hand-write `mlflow.update_current_trace(...)`.

Relevant docs:

- Databricks trace enrichment: <https://docs.databricks.com/aws/en/mlflow3/genai/tracing/attach-tags/>
- Databricks trace UI: <https://docs.databricks.com/aws/en/mlflow3/genai/tracing/observe-with-traces/ui-traces>
- MLflow LangChain autologging: <https://mlflow.org/docs/latest/api_reference/python_api/mlflow.langchain.html>
- MLflow `update_current_trace`: <https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.update_current_trace>

Project docs:

- [docs/ui-attributes-guide.md](docs/ui-attributes-guide.md): how to add new tags, metadata, previews, and dynamic builders for future trace UI display
- [docs/performance-tuning-guide.md](docs/performance-tuning-guide.md): how to reason about MLflow tracing overhead and tune async/root-trace performance

## What this package adds

- Request-scoped `TraceContext` with `session_id`, `user_id`, tags, metadata, and preview builders
- Auto-injection of a LangChain callback so existing `invoke(...)` and `ainvoke(...)` paths can be enriched with minimal call-site change
- Explicit open/close APIs for teams that do not want `with` blocks
- Root-trace helpers for async frameworks when concurrent `ainvoke(...)` enrichment needs tighter control
- Local examples for Ollama, tool calling, verification, and load testing

## When to use which pattern

### Sync apps

Usually enough:

```python
mlflow.langchain.autolog()
enable_mlflow_langchain_enrichment()
```

Then attach request context with `using_trace_context(...)` or `open_trace_context(...)`.

### Async apps

`mlflow.langchain.autolog()` supports `ainvoke(...)`, but for concurrent async workloads where `session_id` and other enriched fields must land reliably on the root trace, the safer pattern is:

- keep `mlflow.langchain.autolog()` enabled for token usage and child spans
- wrap each request with `using_root_trace(...)` or `open_root_trace(...)`
- call unchanged `await chain.ainvoke(...)` inside that scope

This is the recommended pattern for FastAPI-style async handlers.

## Install

```bash
pip install -e .
```

If `mlflow.langchain.autolog()` fails with `ModuleNotFoundError: No module named 'langchain'`, install the top-level `langchain` package, not only `langchain-core`:

```bash
pip install "langchain>=0.3.19,<1.3.0"
```

## Local setup

Start Ollama and a local MLflow server:

```bash
ollama pull nemotron-3-nano
./scripts/start_mlflow_server.sh
```

Optional server overrides:

```bash
MLFLOW_PORT=5001 ./scripts/start_mlflow_server.sh
MLFLOW_WORK_DIR=/tmp/mlflow-demo ./scripts/start_mlflow_server.sh
```

## Public API

```python
from mlflow_langchain_enrichment import (
    TraceContext,
    invoke_with_enrichment,
    ainvoke_with_enrichment,
    auto_trace_llm,
    auto_trace_chain,
    auto_trace_runnable,
    enable_mlflow_langchain_enrichment,
    get_current_trace_context,
    trace_llm,
    trace_llm_call,
    using_trace_context,
    open_trace_context,
    close_trace_context,
    using_root_trace,
    open_root_trace,
    close_root_trace,
)
```

`TraceContext` supports:

- `tags`: trace tags for UI columns and filtering
- `metadata`: trace metadata for search and enrichment
- `span_metadata`: extra LangChain runnable metadata
- `user_id`: mapped to `metadata["mlflow.trace.user"]`
- `session_id`: mapped to `metadata["mlflow.trace.session"]`
- `client_request_id`: external correlation id
- `request_preview` / `response_preview`: explicit trace list text
- `request_preview_builder` / `response_preview_builder`: custom preview logic
- `tags_builder` / `metadata_builder`: derive final trace fields from `(runnable, inputs, response, error)`
- `trace_name`: span or LangChain run name
- `mlflow_run_name`: optional associated MLflow Run name
- `run_tags`: optional tags for the associated MLflow Run
- `run_description`: optional description for the associated MLflow Run
- `ensure_run`: open an MLflow Run if none is active
- `capture_root_span_io`: control whether `using_root_trace(...)` writes root span `Inputs` / `Outputs`

## Minimum-change integration

Enable autologging once at startup:

```python
import mlflow

from mlflow_langchain_enrichment import enable_mlflow_langchain_enrichment

mlflow.langchain.autolog()
enable_mlflow_langchain_enrichment()
```

Then set request context around your existing application flow:

```python
from mlflow_langchain_enrichment import using_trace_context

with using_trace_context(
    user_id=user_id,
    session_id=session_id,
    tags={"app": "support-bot", "route": route_name},
    metadata={"deployment": "prod"},
):
    result = chain.invoke(payload)
```

The `chain.invoke(payload)` call stays unchanged.

If your team prefers explicit lifecycle control:

```python
from mlflow_langchain_enrichment import close_trace_context, open_trace_context

handle = open_trace_context(
    user_id=user_id,
    session_id=session_id,
    mlflow_run_name=f"chat-{session_id}",
    ensure_run=True,
    tags={"app": "support-bot", "route": route_name},
)
try:
    result = chain.invoke(payload)
finally:
    close_trace_context(handle)
```

If you already have request context in framework state, register a provider once:

```python
from mlflow_langchain_enrichment import TraceContext, enable_mlflow_langchain_enrichment


def current_trace_context() -> TraceContext:
    return TraceContext(
        user_id=request_state.user_id,
        session_id=request_state.session_id,
        tags={"app": "support-bot", "route": request_state.route},
        metadata={"deployment": "prod"},
    )


enable_mlflow_langchain_enrichment(current_trace_context)
```

After that, existing `invoke(...)` and `ainvoke(...)` paths pick up enrichment automatically whenever the provider returns a context.

## Async production pattern

For concurrent async handlers, use a root trace per request and keep autologging enabled:

```python
import mlflow

from mlflow_langchain_enrichment import using_root_trace

mlflow.langchain.autolog()


async def traced_request(chain, payload, session_id, user_id, request_id):
    with using_root_trace(
        user_id=user_id,
        session_id=session_id,
        client_request_id=request_id,
        trace_name=f"chat-{session_id}",
        capture_root_span_io=False,
    ) as trace:
        trace.request = payload
        result = await chain.ainvoke(payload)
        trace.response = result
        return result
```

Why this is the recommended async shape:

- root trace gets `session_id`, `user_id`, request/response previews, and request correlation
- LangChain autologging still emits child spans such as `ChatOllama` and tool spans
- token usage remains available through the autologged model span
- `capture_root_span_io=False` keeps the root span as an envelope and avoids duplicating the child model span inputs/outputs

If your team prefers explicit lifecycle control instead of a `with` block, use `open_root_trace(...)` and `close_root_trace(...)`.

## Ergonomic APIs

For simpler call sites, there are three higher-level entrypoints.

### 1. Auto-wrapped LLM

```python
from mlflow_langchain_enrichment import auto_trace_llm

llm = ChatOllama(model="nemotron-3-nano", temperature=0)
llm = auto_trace_llm(
    llm,
    user_id=user_id,
    session_id=session_id,
    trace_name="billing-chat",
    ensure_run=True,
    tags={"app": "support-bot", "model": llm.model},
    metadata={"app_version": "0.1.0", "model_name": llm.model},
    tags_builder=lambda runnable, inputs, response, error: {
        "response_kind": "tool" if getattr(response, "tool_calls", None) else "chat",
    },
    metadata_builder=lambda runnable, inputs, response, error: {
        "response_chars": len(getattr(response, "content", "") or ""),
    },
)

response = await llm.ainvoke(messages)
```

### 2. Context manager

```python
from mlflow_langchain_enrichment import trace_llm

async with trace_llm(
    user_id=user_id,
    session_id=session_id,
    trace_name="billing-chat",
) as trace:
    trace.set_request(messages)
    response = await llm.ainvoke(messages)
    trace.add_tags({"model": llm.model})
    trace.add_metadata({"model_name": llm.model})
    trace.set_response(response)
```

### 3. Decorator

```python
from mlflow_langchain_enrichment import trace_llm_call

@trace_llm_call(
    user_id=user_id,
    session_id=session_id,
    trace_name="billing-chat",
)
async def agent(llm, input):
    return await llm.ainvoke(input)
```

The decorator auto-detects a likely request argument for common signatures such as `func(llm, input)` or `func(input=...)`. If your function shape is different, pass `request_resolver=...`.

If you need to inspect the response first and then attach new UI fields, prefer `trace_llm(...)` over `auto_trace_llm(...)`. `TraceSession` supports `add_tags(...)`, `add_metadata(...)`, and `update_trace(...)` before the trace closes.

## Async limitations and references

MLflow documents async LangChain autologging as supported for `ainvoke`, `abatch`, and `astream`, but it also explicitly warns that the logging work itself is not asynchronous and may block the main thread. That means async tracing can still add latency under load even when the model call is awaited normally.

For concurrent async applications, MLflow also documents `mlflow.tracing.set_destination(..., context_local=True)` as the mechanism for task-local destination isolation. That helps with routing traces per task or thread, but it does not by itself guarantee that every autologged async trace-enrichment pattern will behave correctly in all combinations.

This package recommends `using_root_trace(...)` for production async request handlers because it gives you explicit request-level control over session/user enrichment while still keeping `mlflow.langchain.autolog()` enabled for child spans and token usage.

References:

- MLflow LangChain autologging async warning: <https://mlflow.org/docs/latest/genai/flavors/langchain/autologging/>
- MLflow LangChain API reference, including `run_tracer_inline`: <https://mlflow.org/docs/latest/api_reference/python_api/mlflow.langchain.html>
- MLflow tracing FAQ on `context_local=True`: <https://www.mlflow.org/docs/3.3.0/genai/tracing/faq/>
- MLflow issue `#16880` on async manual-trace + autolog hierarchy problems: <https://github.com/mlflow/mlflow/issues/16880>
- MLflow issue `#18216` on separate traces when combining manual and automatic tracing: <https://github.com/mlflow/mlflow/issues/18216>

## Simple local example

```python
import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import TraceContext, invoke_with_enrichment

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("langchain-trace-enrichment")
mlflow.langchain.autolog()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise support assistant."),
        ("human", "{question}"),
    ]
)
chain = (prompt | ChatOllama(model="nemotron-3-nano", temperature=0)).with_config(
    {"run_name": "support-assistant"}
)

trace_context = TraceContext(
    user_id="user-42",
    session_id="session-20260306-001",
    client_request_id="req-20260306-abc",
    tags={"app": "support-bot", "environment": "dev", "feature": "billing-help"},
    metadata={"app_version": "0.1.0", "deployment": "local-mlflow-server"},
    span_metadata={"tenant": "local-demo", "provider": "ollama"},
)

result = invoke_with_enrichment(
    chain,
    {"question": "Why was invoice INV-42 charged twice?"},
    trace_context,
)

print(result.content)
```

## Tool-calling example

`langchain 1.x` tool-calling with `ChatOllama` is in [examples/local_ollama_tool_calling_mlflow.py](examples/local_ollama_tool_calling_mlflow.py). In the trace UI you should see nested spans for:

- root request trace
- agent execution
- chat model call
- each tool invocation

## Verification and load-test examples

- [examples/local_ollama_mlflow.py](examples/local_ollama_mlflow.py): basic sync enrichment example
- [examples/local_ollama_auto_enrichment_mlflow.py](examples/local_ollama_auto_enrichment_mlflow.py): unchanged `chain.invoke(...)` with automatic context injection
- [examples/local_ollama_trace_llm_patterns.py](examples/local_ollama_trace_llm_patterns.py): `auto_trace_llm(...)`, `trace_llm(...)`, and `trace_llm_call(...)`
- [examples/async_auto_trace_llm_load_test.py](examples/async_auto_trace_llm_load_test.py): 10 concurrent `auto_trace_llm(...)` requests with per-request trace attributes
- [examples/async_auto_trace_llm_stream_test.py](examples/async_auto_trace_llm_stream_test.py): concurrent `auto_trace_llm(...).astream(...)` verification
- [examples/local_ollama_deepagents_supervisor_mlflow.py](examples/local_ollama_deepagents_supervisor_mlflow.py): Deep Agents supervisor with 3 traced subagents sharing one `session_id`
- [examples/local_ollama_supervisor_subagents_mlflow.py](examples/local_ollama_supervisor_subagents_mlflow.py): supervisor agent plus 3 traced subagents sharing one `session_id`
- [examples/local_ollama_tool_calling_mlflow.py](examples/local_ollama_tool_calling_mlflow.py): tool-calling trace example
- [examples/verify_open_close_trace_context.py](examples/verify_open_close_trace_context.py): verify explicit trace-context lifecycle
- [examples/async_load_test_trace_context.py](examples/async_load_test_trace_context.py): concurrent async autolog enrichment test
- [examples/threaded_load_test_trace_context.py](examples/threaded_load_test_trace_context.py): threaded fallback when async autolog context is unreliable
- [examples/async_manual_root_trace_load_test.py](examples/async_manual_root_trace_load_test.py): concurrent async root-trace verification

## What appears in the UI

This package enriches the trace UI as follows:

- `Request`: generated from LangChain input or overridden with `request_preview`
- `Response`: generated from chain output or overridden with `response_preview`
- `Session`: from `metadata["mlflow.trace.session"]`
- `User`: from `metadata["mlflow.trace.user"]`
- custom tag columns such as `app`, `environment`, `feature`
- searchable metadata such as `app_version`, `deployment`, or custom business attributes

Typical queries:

```text
metadata.`mlflow.trace.user` = 'user-42'
metadata.`mlflow.trace.session` = 'session-20260306-001'
metadata.app_version = '0.1.0'
tags.environment = 'dev'
```

If the UI `Run name` column is empty, the trace is not associated with an active MLflow Run. Use `ensure_run=True` or wrap the request in your own `mlflow.start_run(...)` context.

## Files

- package code: `src/mlflow_langchain_enrichment/`
- examples: `examples/`
- server script: `scripts/start_mlflow_server.sh`
- tests: `tests/test_enrichment.py`

# mlflow-langchain-enrichment

Small Python package for enriching `mlflow.langchain.autolog()` traces with better trace list display and searchable trace context.

## Why this exists

The Databricks trace UI can show:

- `Request` and `Response` previews
- `Session` and `User`
- custom `Tags` as columns
- trace `metadata` that you can filter on

The missing piece with LangChain autolog is that you still need to call `mlflow.update_current_trace(...)` while the autologged trace is active. This package does that by attaching a small LangChain callback handler to the same invocation that autolog instruments.

Relevant docs:

- Databricks trace enrichment: <https://docs.databricks.com/aws/en/mlflow3/genai/tracing/attach-tags/>
- Databricks trace UI: <https://docs.databricks.com/aws/en/mlflow3/genai/tracing/observe-with-traces/ui-traces>
- MLflow `update_current_trace`: <https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.update_current_trace>
- MLflow `mlflow.langchain.autolog()` compatibility: <https://mlflow.org/docs/latest/api_reference/python_api/mlflow.langchain.html>

## Install

```bash
pip install -e .
```

If `mlflow.langchain.autolog()` raises `ModuleNotFoundError: No module named 'langchain'`, install the top-level `langchain` package, not just `langchain-core`:

```bash
pip install "langchain>=0.3.19,<1.3.0"
```

This repo now declares that dependency in `pyproject.toml`.

Local example prerequisites:

```bash
ollama pull nemotron-3-nano
./scripts/start_mlflow_server.sh
```

Script options:

```bash
MLFLOW_PORT=5001 ./scripts/start_mlflow_server.sh
MLFLOW_WORK_DIR=/tmp/mlflow-demo ./scripts/start_mlflow_server.sh
```

## Package API

```python
from mlflow_langchain_enrichment import TraceContext, invoke_with_enrichment
```

`TraceContext` supports:

- `tags`: mutable trace tags for UI columns and filtering
- `metadata`: immutable trace metadata
- `user_id`: mapped to `metadata["mlflow.trace.user"]`
- `session_id`: mapped to `metadata["mlflow.trace.session"]`
- `client_request_id`: external request correlation id
- `request_preview`: explicit `Request` column text
- `response_preview`: explicit `Response` column text
- `request_preview_builder` / `response_preview_builder`: custom preview logic
- `span_metadata`: extra LangChain `RunnableConfig.metadata`
- `trace_name`: sets LangChain `run_name` when one is not already provided

## Local MLflow + Ollama example

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
    tags={
        "app": "support-bot",
        "environment": "dev",
        "feature": "billing-help",
    },
    metadata={
        "app_version": "0.1.0",
        "deployment": "local-mlflow-server",
    },
    span_metadata={"tenant": "local-demo", "provider": "ollama"},
)

result = invoke_with_enrichment(
    chain,
    {"question": "Why was invoice INV-42 charged twice?"},
    trace_context,
    config={"metadata": {"route": "billing"}},
)

print(result.content)
```

The full runnable script is in [examples/local_ollama_mlflow.py](/Users/binzhang/vibe_coding_repo/mlflow-features/examples/local_ollama_mlflow.py).

## Tool-calling trace example

```python
import mlflow
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from mlflow_langchain_enrichment import TraceContext, invoke_with_enrichment


@tool
def lookup_account_tier(account_id: str) -> str:
    """Look up the support tier for an account."""
    tiers = {"ACME-42": "enterprise", "STARTUP-7": "pro"}
    return tiers.get(account_id, "standard")


@tool
def lookup_ticket_status(ticket_id: str) -> str:
    """Look up the current status of a support ticket."""
    statuses = {"TICK-1001": "Open and assigned to billing."}
    return statuses.get(ticket_id, "Ticket not found.")


agent = create_agent(
    model=ChatOllama(model="nemotron-3-nano", temperature=0),
    tools=[lookup_account_tier, lookup_ticket_status],
    system_prompt="Use tools for account and ticket questions.",
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("langchain-trace-enrichment")
mlflow.langchain.autolog()

result = invoke_with_enrichment(
    agent,
    {
        "messages": [
            {
                "role": "user",
                "content": "What tier is account ACME-42 on and what is ticket TICK-1001 status?",
            }
        ]
    },
    TraceContext(tags={"feature": "tool-calling"}),
)

print(result["messages"][-1].content)
```

The full runnable script is in [examples/local_ollama_tool_calling_mlflow.py](/Users/binzhang/vibe_coding_repo/mlflow-features/examples/local_ollama_tool_calling_mlflow.py).

This example targets `langchain 1.x`, which uses `create_agent(...)` instead of the older `AgentExecutor` / `create_tool_calling_agent(...)` API.

In the trace UI you should see nested spans for the agent execution, the chat model call, and each tool invocation. This is an inference from the official MLflow LangChain autolog docs plus the LangChain 1.x agent flow:

- MLflow autolog traces nested LangChain callbacks: <https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain/>
- LangChain agent API: <https://docs.langchain.com/oss/python/langchain/agents>

## What shows up in the MLflow / Databricks UI

- `Request`: built from the LangChain input, or overridden with `request_preview`
- `Response`: built from the final chain output, or overridden with `response_preview`
- `Session`: populated from `metadata["mlflow.trace.session"]`
- `User`: populated from `metadata["mlflow.trace.user"]`
- tag columns such as `app`, `environment`, `feature`
- metadata filters such as `metadata.app_version = '0.1.0'`

In the Traces tab, use `Columns` to add tag columns and use queries like:

```text
metadata.`mlflow.trace.user` = 'user-42'
metadata.`mlflow.trace.session` = 'session-20260306-001'
metadata.app_version = '0.1.0'
tags.environment = 'dev'
```

## Files

- package code: `src/mlflow_langchain_enrichment/`
- example: `examples/local_ollama_mlflow.py`
- tool-calling example: `examples/local_ollama_tool_calling_mlflow.py`
- mlflow server script: `scripts/start_mlflow_server.sh`
- tests: `tests/test_enrichment.py`

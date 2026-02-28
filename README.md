# mlflow-features

Custom MLflow tracing adapter for LangChain-style agent telemetry when `langchain autolog` is not compatible with a custom gateway/client stack.

## Environment
- Python: `3.12`
- Package manager: `uv`
- Core model path for this project: `ChatOllama(model="glm-4.7-flash")`

## Install
```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv sync
```

## Configure MLflow
Set environment variables as needed:
```bash
export AGENT_TELEMETRY_ENABLED=true
export AGENT_TELEMETRY_MLFLOW_TRACKING_URI=http://localhost:5000
export AGENT_TELEMETRY_MLFLOW_EXPERIMENT=agent-observability
export AGENT_TELEMETRY_SERVICE_NAME=my-agent-service
export AGENT_TELEMETRY_SERVICE_VERSION=0.1.0
export AGENT_TELEMETRY_ENVIRONMENT=dev
```

## Register Callback (LangChain Path)
```python
from langchain_ollama import ChatOllama

from agent_mlflow_telemetry import (
    TelemetryConfig,
    build_langchain_callback,
    initialize_telemetry,
)

config = TelemetryConfig.from_env()
runtime = initialize_telemetry(config)
callback = build_langchain_callback(runtime)

model = ChatOllama(model="glm-4.7-flash", temperature=0)
response = model.invoke("hello", config={"callbacks": [callback]})
```

## Adapt `llmclient` as ChatModel
```python
from langchain_ollama import ChatOllama

from agent_mlflow_telemetry import TelemetryConfig, initialize_telemetry, wrap_llmclient

runtime = initialize_telemetry(TelemetryConfig.from_env())
client = ChatOllama(model="glm-4.7-flash")
instrumented = wrap_llmclient(client, runtime.sink)

result = instrumented.invoke("hello")
```

## Run Reference Example
```bash
UV_CACHE_DIR=.uv-cache uv run python examples/chat_ollama_reference.py
```

If local Ollama is not running or `glm-4.7-flash` is not pulled, the example prints an actionable error instead of crashing silently.

## Telemetry Behaviors
- Span lifecycle: start/end/event write path through `MLflowSink`
- Retry: bounded retry on transient failures
- Failure mode: fail-open by default (telemetry failures do not break agent execution)
- Required schema fields enforced (`trace_id`, `span_id`, `component`, `operation`, `status`)

## Troubleshooting
- Empty attribute columns:
  - Ensure spans are built via `SpanRecord` and `to_attributes()` only.
  - Avoid overriding required keys in `attributes`.
- Broken parent/child hierarchy:
  - Ensure callback methods receive `run_id` and `parent_run_id` from LangChain runtime.
  - Use `with_trace_context(...)` to propagate upstream trace/session IDs into root spans.
- Missing model/provider fields:
  - Pass model metadata in invocation params or use clients with standard fields like `model`/class-name provider hints.

## Run Tests
```bash
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

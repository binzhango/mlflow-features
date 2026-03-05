# mlflow-features

Custom MLflow tracing adapter for LangChain-style agent telemetry when `langchain autolog` is not compatible with a custom gateway/client stack.

## Environment
- Python: `3.12`
- Package manager: `uv`

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

## Run MLflow Server
```bash
chmod +x scripts/run_mlflow_server.sh
./scripts/run_mlflow_server.sh
```

Optional overrides:
```bash
MLFLOW_HOST=0.0.0.0 MLFLOW_PORT=5001 ./scripts/run_mlflow_server.sh
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

model = ChatOllama(model="nemotron-3-nano", temperature=0)
response = model.invoke("hello", config={"callbacks": [callback]})
```

## Autolog-Style Global Tracer (No Per-Invoke Callback)
```python
from langchain_ollama import ChatOllama

from agent_mlflow_telemetry import TelemetryConfig, autolog

runtime = autolog(config=TelemetryConfig.from_env())

model = ChatOllama(model="nemotron-3-nano", temperature=0)
response = model.invoke("hello")
```

If needed, disable global injection later:

```python
from agent_mlflow_telemetry import disable_autolog

disable_autolog()
```

## Adapt `llmclient` as ChatModel
```python
from langchain_ollama import ChatOllama

from agent_mlflow_telemetry import TelemetryConfig, initialize_telemetry, wrap_llmclient

runtime = initialize_telemetry(TelemetryConfig.from_env())
client = ChatOllama(model="nemotron-3-nano", temperature=0)
instrumented = wrap_llmclient(client, runtime.sink)

result = instrumented.invoke("hello")
```

## Run Reference Example
```bash
UV_CACHE_DIR=.uv-cache uv run python examples/chat_ollama_reference.py
```

If local Ollama is not running or `nemotron-3-nano` is not pulled, the example prints an actionable error instead of crashing silently.

## Run Autolog Verification Example
This example uses `ChatOllama` and verifies `CHAT_MODEL` traces for both `invoke()` and `ainvoke()` without `mlflow.start_run()`:

```bash
UV_CACHE_DIR=.uv-cache uv run python examples/autolog_verification.py
```

Optional model override:

```bash
OLLAMA_MODEL=nemotron-3-nano UV_CACHE_DIR=.uv-cache uv run python examples/autolog_verification.py
```

## Run Tool Autolog Verification Example
This example binds a tool to `ChatOllama`, runs in autolog mode, and checks that `tool_call.*` spans are present:

```bash
OLLAMA_MODEL=nemotron-3-nano UV_CACHE_DIR=.uv-cache uv run python examples/tool_autolog_verification.py
```

## Run Tool-Call UI Demo
This demo does not require a live LLM. It emits a deterministic trace with a child tool-call span:

```bash
AGENT_TELEMETRY_MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
UV_CACHE_DIR=.uv-cache uv run python examples/tool_call_span_ui_demo.py
```

After it runs, verify in MLflow UI that:
- a child span named `tool_call.function.lookup_weather` is present under the first LLM span.
- the LLM span attributes include `model_config.temperature` and `model_config.reasoning_effort`.

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

## Guides
- Span/UI customization guide: [`docs/customize-trace-spans-ui.md`](docs/customize-trace-spans-ui.md)

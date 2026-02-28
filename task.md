# Task: Implement Custom MLflow Telemetry Adapter (Python 3.12 + uv)

## Goal
Implement core telemetry features for agent usage tracking without `langchain autolog`, using a custom MLflow adapter.

## Constraints Locked
- Runtime: Python 3.12
- Package/tooling: `uv`
- LLM integration model for core implementation: `ChatOllama`
- Model name: `glm-4.7-flash`
- Treat `llmclient` as a LangChain-style `ChatModel` abstraction (similar to `ChatOpenAI` / `ChatOllama`)
- Telemetry sink: MLflow Tracing API
- Failure mode: fail-open (never break agent flow on telemetry errors)

## Deliverables
1. Python package: `agent_mlflow_telemetry/`
2. LangChain callback handler for span lifecycle
3. `llmclient` wrapper API that assumes ChatModel-like interface
4. Working reference integration using `ChatOllama(model="glm-4.7-flash")`
5. Tests (unit + integration)
6. Minimal usage docs

## Implementation Tasks

### 1. Bootstrap project environment
- [x] Create/verify `.python-version` set to `3.12`
- [x] Initialize env with `uv`
- [x] Add dependencies:
  - `mlflow`
  - `langchain-core`
  - `langchain-ollama`
  - `pydantic` (or dataclass-only if preferred)
  - testing deps: `pytest`, `pytest-asyncio`
- [x] Add `Makefile` or scripts for:
  - setup
  - test
  - lint (optional)

### 2. Define telemetry package structure
- [x] Create package layout:
  - `agent_mlflow_telemetry/config.py`
  - `agent_mlflow_telemetry/context.py`
  - `agent_mlflow_telemetry/schema.py`
  - `agent_mlflow_telemetry/mlflow_sink.py`
  - `agent_mlflow_telemetry/langchain_callback.py`
  - `agent_mlflow_telemetry/llmclient_adapter.py`
  - `agent_mlflow_telemetry/bootstrap.py`

### 3. Config and context propagation
- [x] Implement `TelemetryConfig` with required fields
- [x] Implement context propagation using `contextvars`:
  - `trace_id`, `span_id`, `parent_span_id`
  - `session_id`, `root_request_id`
- [x] Add helper `with_trace_context(...)`

### 4. Canonical schema mapping
- [x] Implement normalized span attribute builder with required fields:
  - identity/hierarchy
  - component/operation
  - model/gateway metadata
  - latency/tokens
  - status/error fields
- [x] Ensure strict handling of required vs optional attributes

### 5. MLflow sink
- [x] Implement span lifecycle writer:
  - `start_span(...)`
  - `end_span(...)`
  - `record_event(...)`
- [x] Add retry-on-transient failure (bounded retries)
- [x] Add fail-open behavior with warning logs

### 6. LangChain callback handler
- [x] Implement handler hooks:
  - chain start/end/error
  - tool start/end/error
  - llm start/end/error
  - retriever start/end/error (if used)
- [x] Map all hooks to canonical schema
- [x] Ensure parent-child span relationships are consistent

### 7. `llmclient` ChatModel-style adapter
- [x] Implement wrapper expecting ChatModel-like methods:
  - sync invoke
  - async invoke
  - streaming
- [x] Preserve interface compatibility so existing `llmclient` usage can be adapted with minimal changes
- [x] Add model metadata capture compatible with:
  - `ChatOllama(model="glm-4.7-flash")`

### 8. Core reference implementation with ChatOllama
- [x] Add runnable example script:
  - instantiate telemetry runtime
  - create callback
  - create `ChatOllama(model="glm-4.7-flash")`
  - run one simple agent/chain flow
  - verify trace written to MLflow

### 9. Tests
- [x] Unit tests for:
  - context propagation
  - schema completeness
  - fail-open sink behavior
  - sync/async/stream wrapper behavior
- [x] Integration tests for:
  - valid trace tree generation
  - token/latency attribute population
  - error path span correctness
- [x] Regression test for malformed/empty attributes issue

### 10. Documentation
- [x] Add `README` section:
  - install with `uv`
  - configure MLflow
  - register callback
  - adapt `llmclient`
  - run ChatOllama example
- [x] Add troubleshooting notes for empty attributes and broken span hierarchy

## Acceptance Criteria
- [x] Root span is created per agent request
- [x] Child spans for chain/tool/llm are correctly linked
- [x] No empty required fields in emitted telemetry
- [x] Sync + async + streaming paths all instrumented
- [x] Telemetry write failures do not break application flow
- [x] Reference path works with `ChatOllama` + `glm-4.7-flash`

## Suggested Commands (uv + Python 3.12)
```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv add mlflow langchain-core langchain-ollama pydantic
uv add --dev pytest pytest-asyncio
uv run pytest -q
```

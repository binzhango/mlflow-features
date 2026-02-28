<proposed_plan>
# Plan: Custom MLflow Telemetry Adapter for LangChain + `llmclient` (Python)

## Summary
Build a reusable Python package that bypasses `langchain autolog` and logs agent activity directly to MLflow Tracing using a custom LangChain callback handler plus a thin `llmclient` adapter.  
The design will support sync/async/streaming calls, preserve upstream trace context, and run in fail-open mode so telemetry failures never block agent execution.  
Based on your choices, the default policy is full prompt/response text logging with 100% sampling in all environments.

## Scope
- In scope:
  - Python + LangChain integration
  - MLflow Tracing as primary log sink
  - Callback-based instrumentation for agent/chain/tool/LLM events
  - `llmclient` wrapper hooks for normalized request/response metadata
  - Internal reusable package with registration/config helpers
- Out of scope (v1):
  - TypeScript runtime
  - Migration of old traces
  - UI customization in MLflow front-end
  - Cross-vendor OTEL exporter integration (can be v2)

## Architecture
1. `telemetry_core` module:
   - Trace/span lifecycle management
   - Context propagation (upstream trace ID, session ID, user/request IDs)
   - Sampling gate (configured at 100% initially)
2. `mlflow_sink` module:
   - Writes spans/events/attributes to MLflow Tracing
   - Handles retries + fail-open error handling
3. `langchain_callback` module:
   - Custom callback handler implementing chain/tool/agent/LLM hooks
   - Maps LangChain events to canonical span schema
4. `llmclient_adapter` module:
   - Wraps `llmclient` sync/async/streaming calls
   - Captures model name, gateway route, token usage, latency, status/error
5. `config` module:
   - Env-driven config (tracking URI, experiment, sampling, content logging flags)
   - Runtime toggles and guardrails
6. `bootstrap` module:
   - One-call registration helper for service startup
   - Attaches callback handler + optional middleware for root trace context

## Public Interfaces / API Changes
Create an internal package, e.g. `agent_mlflow_telemetry`, with the following public surfaces:

1. `TelemetryConfig` (dataclass/Pydantic model)
   - `enabled: bool`
   - `mlflow_tracking_uri: str | None`
   - `mlflow_experiment: str | None`
   - `sampling_rate: float` (default `1.0`)
   - `log_content: bool` (default `true`, per your preference)
   - `fail_open: bool` (default `true`)
   - `service_name: str`
   - `service_version: str | None`
   - `environment: str`
2. `initialize_telemetry(config: TelemetryConfig) -> TelemetryRuntime`
   - Initializes MLflow client/sink and context manager
3. `build_langchain_callback(runtime: TelemetryRuntime) -> BaseCallbackHandler`
   - Returns callback handler to inject into chains/agents
4. `wrap_llmclient(client, runtime: TelemetryRuntime) -> InstrumentedLLMClient`
   - Decorator/wrapper preserving original method signatures
   - Supports sync/async/streaming calls
5. `with_trace_context(...)` utility
   - Accepts upstream IDs and request metadata
   - Ensures root trace initialization and propagation

## Canonical Telemetry Schema (Decision-Complete)
All spans/events should emit these normalized attributes (required unless stated optional):

- Identity and hierarchy:
  - `trace_id` (required)
  - `span_id` (required)
  - `parent_span_id` (optional for root)
  - `root_request_id` (optional)
  - `session_id` (optional)
  - `user_id` (optional, if policy permits)
- Runtime metadata:
  - `service.name`, `service.version`, `env`
  - `component` (`agent|chain|tool|llm|retriever`)
  - `operation` (normalized action name)
- Model/gateway:
  - `provider`
  - `model_name`
  - `gateway_route`
  - `endpoint` (optional sanitized URI path)
- Usage/perf:
  - `latency_ms`
  - `input_tokens` (optional if unavailable)
  - `output_tokens` (optional if unavailable)
  - `total_tokens` (optional)
  - `streamed` (`true|false`)
- Content and payload:
  - `prompt_text` (logged in v1, per your policy)
  - `response_text` (logged in v1)
  - `prompt_hash` (still include for dedupe/search)
- Outcome:
  - `status` (`ok|error|cancelled|timeout`)
  - `error_type` (optional)
  - `error_message` (optional, truncated)
  - `http_status` (optional)

## Span Mapping Rules
1. Root:
   - One root span per agent request
   - Root IDs sourced from upstream context when available; fallback to generated IDs
2. Agent/chain:
   - `on_chain_start` -> open span
   - `on_chain_end` -> close span with status `ok`
   - `on_chain_error` -> close span with status `error`
3. Tool:
   - `on_tool_start` / `on_tool_end` / `on_tool_error` mapped similarly
4. LLM:
   - `on_llm_start` -> child span
   - `on_llm_end` -> attach token/latency/model attrs and close
   - `on_llm_error` -> error span close
5. Streaming:
   - One LLM span per request, optional chunk events attached to same span
   - Finalization occurs on stream completion/error
6. Retries:
   - Represent each retry as child attempt span or attempt counter attribute
   - Preserve single logical parent LLM operation span

## Failure Handling and Reliability
- Fail-open behavior:
  - Any telemetry write exception must not raise into business logic
  - Emit local warning log with concise error code/message
- Retry strategy:
  - Lightweight retry for transient MLflow failures (bounded attempts, short backoff)
  - If all retries fail, drop span and continue
- Circuit guard:
  - Optional in-memory cooldown after repeated sink failures to reduce overhead

## Integration Plan (Implementation Phases)
1. Package scaffold
   - Create `agent_mlflow_telemetry/` with modules above
   - Add dependency pins and extras for `mlflow`, `langchain-core`
2. Core context + schema
   - Implement context var storage for trace/span IDs
   - Implement span attribute normalizer + validators
3. MLflow sink
   - Implement span start/end/event write functions
   - Add fail-open wrappers and retry policy
4. LangChain callback handler
   - Implement full hook coverage for chain/tool/llm/retriever/error paths
5. `llmclient` wrapper
   - Add sync/async/streaming interception
   - Ensure no signature-breaking changes
6. Bootstrap utilities
   - Provide registration helper for app startup and per-request middleware helper
7. Documentation
   - Add integration examples: minimal agent, async agent, streaming agent
   - Add troubleshooting table for empty attributes/malformed spans

## Testing Strategy
1. Unit tests
   - Context propagation (upstream IDs, nested spans, fallback generation)
   - Schema mapping correctness for each hook
   - Sync/async/streaming wrapper behavior
   - Fail-open behavior under MLflow exceptions
2. Contract tests
   - Validate emitted span payload fields and required attributes
   - Ensure no empty required fields
3. Integration tests
   - End-to-end LangChain agent run with mocked `llmclient` and MLflow sink
   - Error path tests: tool failure, LLM timeout, streaming interruption
4. Regression tests
   - Reproduce current malformed/empty attribute scenarios and verify fix
5. Load sanity
   - High-concurrency async requests to verify context isolation and overhead bounds

## Acceptance Criteria
- 100% of agent requests create a valid root span
- Parent/child relationships are consistent for chain/tool/LLM spans
- Required schema fields are populated (no empty required columns)
- Sync/async/streaming `llmclient` paths are all instrumented
- Telemetry failures never break agent execution (fail-open)
- Trace IDs match upstream request context when supplied

## Rollout and Monitoring
1. Stage rollout
   - Enable in one pilot service with 100% sampling
2. Observe
   - Monitor dropped span count, sink error rate, callback overhead, span completeness
3. Harden
   - Tune retry/cooldown and attribute truncation limits based on pilot
4. Expand
   - Roll package to additional services with same config contract

## Assumptions and Defaults Locked
- Runtime: Python + LangChain
- Sink: MLflow Tracing API
- Instrumentation: LangChain callback handler + `llmclient` wrapper
- Logging content: full prompt/response text enabled by default
- Sampling: 100% all environments
- Trace ID source: upstream request context preferred, generated fallback
- Failure mode: fail-open
- Delivery model: reusable internal package
</proposed_plan>

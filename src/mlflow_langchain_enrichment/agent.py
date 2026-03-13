from __future__ import annotations

import contextvars
import copy
from dataclasses import dataclass
from functools import wraps
from typing import Any, AsyncIterator, Iterator, Mapping

from .auto import accumulate_current_trace_token_usage, close_traced_span, open_traced_span
from .enrichment import TraceContext

try:
    from langgraph.graph.state import StateGraph
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.graph.state import StateNodeSpec
    from langgraph.types import RetryPolicy
except ImportError:  # pragma: no cover - optional dependency at import time
    StateGraph = None  # type: ignore[assignment]
    CompiledStateGraph = None  # type: ignore[assignment]
    StateNodeSpec = None  # type: ignore[assignment]
    RetryPolicy = None  # type: ignore[assignment]


TraceContextLike = TraceContext | Mapping[str, Any] | None
_CHAT_TOKEN_USAGE_ATTRIBUTE_KEY = "mlflow.chat.tokenUsage"
_LLM_MODEL_ATTRIBUTE_KEY = "mlflow.llm.model"
_TOKEN_USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")
_AGENT_SPAN_TYPE = "AGENT"
_CURRENT_AGENT_EXECUTION: contextvars.ContextVar["AgentExecutionContext | None"] = (
    contextvars.ContextVar(
        "mlflow_langchain_enrichment_agent_execution",
        default=None,
    )
)
_PENDING_CHILD_AGENT: contextvars.ContextVar["PendingChildAgentContext | None"] = (
    contextvars.ContextVar(
        "mlflow_langchain_enrichment_pending_child_agent",
        default=None,
    )
)
_CURRENT_AGENT_SPAN_STACK: contextvars.ContextVar[tuple["AgentSpanBinding", ...]] = (
    contextvars.ContextVar(
        "mlflow_langchain_enrichment_agent_span_stack",
        default=(),
    )
)


@dataclass(slots=True)
class AgentExecutionContext:
    agent_name: str
    agent_type: str
    tool_call_id: str | None = None


@dataclass(slots=True)
class PendingChildAgentContext:
    agent_name: str
    agent_type: str
    tool_call_id: str | None = None


@dataclass(slots=True)
class AgentSpanBinding:
    span: Any
    model_names: list[str]
    token_usage: dict[str, int]
    last_output: Any = None


def _coerce_trace_context(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> TraceContext:
    if trace_context is not None and trace_context_kwargs:
        raise ValueError("Pass either a TraceContext object or keyword fields, not both.")

    if trace_context is None:
        return TraceContext(**trace_context_kwargs)
    if isinstance(trace_context, TraceContext):
        return trace_context
    if isinstance(trace_context, Mapping):
        return TraceContext(**dict(trace_context))
    raise TypeError(
        "Trace context must be a TraceContext, a mapping of TraceContext fields, or None."
    )


def _resolve_agent_name(
    runnable: Any,
    trace_context: TraceContext,
    explicit_name: str | None = None,
) -> str:
    if explicit_name:
        return explicit_name
    if trace_context.trace_name:
        return trace_context.trace_name

    name = getattr(runnable, "name", None)
    if isinstance(name, str) and name:
        return name

    get_name = getattr(runnable, "get_name", None)
    if callable(get_name):
        try:
            resolved = get_name()
        except Exception:  # pragma: no cover - defensive
            resolved = None
        if isinstance(resolved, str) and resolved:
            return resolved

    return type(runnable).__name__


def _merge_config(
    base_config: Mapping[str, Any] | None,
    request_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base_config or {})
    if not request_config:
        return merged

    for key, value in request_config.items():
        if (
            key == "metadata"
            and isinstance(merged.get(key), Mapping)
            and isinstance(value, Mapping)
        ):
            metadata = dict(merged[key])
            metadata.update(value)
            merged[key] = metadata
            continue
        merged[key] = value

    return merged


def _trace_context_for_child_agents(trace_context: TraceContext) -> TraceContext:
    return TraceContext(
        tags=dict(trace_context.tags),
        metadata=dict(trace_context.metadata),
        span_metadata=dict(trace_context.span_metadata),
        run_tags=dict(trace_context.run_tags),
        user_id=trace_context.user_id,
        session_id=trace_context.session_id,
        client_request_id=trace_context.client_request_id,
        mlflow_run_name=trace_context.mlflow_run_name,
        run_description=trace_context.run_description,
        ensure_run=trace_context.ensure_run,
        request_preview=trace_context.request_preview,
        response_preview=trace_context.response_preview,
        request_preview_builder=trace_context.request_preview_builder,
        response_preview_builder=trace_context.response_preview_builder,
        tags_builder=trace_context.tags_builder,
        metadata_builder=trace_context.metadata_builder,
        preview_limit=trace_context.preview_limit,
        trace_name=None,
        capture_root_span_io=trace_context.capture_root_span_io,
    )


def _set_span_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    filtered = {str(key): value for key, value in attributes.items() if value is not None}
    if not filtered:
        return
    if hasattr(span, "set_attributes"):
        span.set_attributes(filtered)
        return
    if hasattr(span, "set_attribute"):
        for key, value in filtered.items():
            span.set_attribute(key, value)


def _set_span_type(span: Any, span_type: str) -> None:
    if hasattr(span, "set_span_type"):
        span.set_span_type(span_type)


def _resolve_model_name(
    *,
    trace_context: TraceContext,
    runnable: Any = None,
    response: Any = None,
) -> str | None:
    candidates = [
        getattr(runnable, "model", None),
        getattr(getattr(runnable, "bound", None), "model", None),
        trace_context.metadata.get("model_name"),
        trace_context.tags.get("model"),
        getattr(response, "response_metadata", None),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
        if isinstance(candidate, Mapping):
            model_name = candidate.get("model")
            if isinstance(model_name, str) and model_name:
                return model_name

    return None


def _extract_agent_message(response: Any) -> Any:
    if isinstance(response, Mapping):
        messages = response.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return message
                if isinstance(message, Mapping):
                    mapped_content = message.get("content")
                    if isinstance(mapped_content, str):
                        return message
        for key in ("output", "answer", "result", "content", "text"):
            value = response.get(key)
            if value is not None:
                return value
    return response


def _normalize_message_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        normalized = dict(value)
        content = normalized.get("content")
        if isinstance(content, str):
            payload = {"content": content}
            if isinstance(normalized.get("response_metadata"), Mapping):
                payload["response_metadata"] = dict(normalized["response_metadata"])
            if isinstance(normalized.get("usage_metadata"), Mapping):
                payload["usage_metadata"] = dict(normalized["usage_metadata"])
            if normalized.get("tool_calls") is not None:
                payload["tool_calls"] = normalized["tool_calls"]
            return payload
        for key in ("output", "answer", "result", "text"):
            if isinstance(normalized.get(key), str):
                return {"content": normalized[key]}
        return normalized

    content = getattr(value, "content", None)
    if isinstance(content, str):
        payload: dict[str, Any] = {"content": content}
        response_metadata = getattr(value, "response_metadata", None)
        usage_metadata = getattr(value, "usage_metadata", None)
        tool_calls = getattr(value, "tool_calls", None)
        if isinstance(response_metadata, Mapping):
            payload["response_metadata"] = dict(response_metadata)
        if isinstance(usage_metadata, Mapping):
            payload["usage_metadata"] = dict(usage_metadata)
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
        return payload

    if isinstance(value, str):
        return {"content": value}
    return value


def _build_agent_span_output(result: Any, binding: AgentSpanBinding | None) -> Any:
    if result is None:
        if binding is None:
            return None
        return _normalize_message_payload(binding.last_output)

    if not isinstance(result, Mapping):
        return result

    enriched = dict(result)
    message_payload = _normalize_message_payload(_extract_agent_message(result))
    if not isinstance(message_payload, Mapping) and binding is not None:
        message_payload = _normalize_message_payload(binding.last_output)

    if isinstance(message_payload, Mapping):
        for key in (
            "content",
            "response_metadata",
            "usage_metadata",
            "tool_calls",
            "additional_kwargs",
            "name",
            "id",
        ):
            if key in message_payload and key not in enriched:
                enriched[key] = message_payload[key]

    if binding is not None and "response_metadata" not in enriched and binding.model_names:
        enriched["response_metadata"] = {"model": binding.model_names[0]}
    if binding is not None and "usage_metadata" not in enriched and binding.token_usage:
        enriched["usage_metadata"] = dict(binding.token_usage)
    return enriched


def _coerce_usage_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}

    input_tokens = _coerce_usage_int(
        value.get("input_tokens", value.get("prompt_eval_count"))
    )
    output_tokens = _coerce_usage_int(
        value.get("output_tokens", value.get("eval_count"))
    )
    total_tokens = _coerce_usage_int(value.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    usage: dict[str, int] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return usage


def _extract_token_usage(response: Any) -> dict[str, int]:
    if response is None:
        return {}

    if isinstance(response, Mapping):
        usage = _extract_usage_mapping(response.get("usage_metadata"))
        response_usage = _extract_usage_mapping(response.get("response_metadata"))
        direct_usage = _extract_usage_mapping(response)
        merged = {**response_usage, **usage, **direct_usage}
        return {key: merged[key] for key in _TOKEN_USAGE_KEYS if key in merged}

    if isinstance(response, (list, tuple)):
        for item in reversed(response):
            usage = _extract_token_usage(item)
            if usage:
                return usage
        return {}

    message = getattr(response, "message", None)
    if message is not None and message is not response:
        usage = _extract_token_usage(message)
        if usage:
            return usage

    generations = getattr(response, "generations", None)
    if isinstance(generations, list) and generations:
        for item in reversed(generations):
            usage = _extract_token_usage(item)
            if usage:
                return usage

    usage = _extract_usage_mapping(getattr(response, "usage_metadata", None))
    response_usage = _extract_usage_mapping(getattr(response, "response_metadata", None))
    merged = {**response_usage, **usage}
    return {key: merged[key] for key in _TOKEN_USAGE_KEYS if key in merged}


def _push_current_agent_span(span: Any) -> contextvars.Token[tuple["AgentSpanBinding", ...]]:
    current = _CURRENT_AGENT_SPAN_STACK.get()
    return _CURRENT_AGENT_SPAN_STACK.set(
        (*current, AgentSpanBinding(span=span, model_names=[], token_usage={}, last_output=None))
    )


def _reset_current_agent_span(token: contextvars.Token[tuple["AgentSpanBinding", ...]]) -> None:
    _CURRENT_AGENT_SPAN_STACK.reset(token)


def has_current_agent_span() -> bool:
    return bool(_CURRENT_AGENT_SPAN_STACK.get())


def record_current_agent_model_observation(
    *,
    trace_context: TraceContext,
    runnable: Any = None,
    response: Any = None,
) -> None:
    current = _CURRENT_AGENT_SPAN_STACK.get()
    if not current:
        return

    binding = current[-1]
    model_name = _resolve_model_name(
        trace_context=trace_context,
        runnable=runnable,
        response=response,
    )
    if model_name and model_name not in binding.model_names:
        binding.model_names.append(model_name)

    usage = _extract_token_usage(response)
    accumulate_current_trace_token_usage(usage)
    for key, value in usage.items():
        binding.token_usage[key] = binding.token_usage.get(key, 0) + value
    binding.last_output = _normalize_message_payload(response)

    attributes: dict[str, Any] = {}
    if binding.model_names:
        attributes["model_name"] = binding.model_names[0]
        attributes[_LLM_MODEL_ATTRIBUTE_KEY] = binding.model_names[0]
        if len(binding.model_names) > 1:
            attributes["model_names"] = ", ".join(binding.model_names)
    if binding.token_usage:
        attributes[_CHAT_TOKEN_USAGE_ATTRIBUTE_KEY] = dict(binding.token_usage)
        for key, value in binding.token_usage.items():
            attributes[key] = value

    _set_span_attributes(binding.span, attributes)


def _extract_task_subagent_graphs(func: Any) -> dict[str, Any] | None:
    closure = getattr(func, "__closure__", None)
    code = getattr(func, "__code__", None)
    if closure is None or code is None:
        return None

    for name, cell in zip(code.co_freevars, closure, strict=False):
        if name == "subagent_graphs" and isinstance(cell.cell_contents, dict):
            return cell.cell_contents
    return None


def _wrap_task_tool_for_pending_child_context(tool: Any) -> None:
    original_sync = getattr(tool, "func", None)
    original_async = getattr(tool, "coroutine", None)

    def _set_pending_context(
        *,
        subagent_type: str | None,
        runtime: Any,
    ) -> contextvars.Token[PendingChildAgentContext | None] | None:
        if not isinstance(subagent_type, str) or not subagent_type:
            return None

        tool_call_id = getattr(runtime, "tool_call_id", None)
        token = _PENDING_CHILD_AGENT.set(
            PendingChildAgentContext(
                agent_name=subagent_type,
                agent_type="subagent",
                tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
            )
        )
        return token

    def _extract_call_details(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str | None, Any]:
        subagent_type = kwargs.get("subagent_type")
        runtime = kwargs.get("runtime")
        if subagent_type is None and len(args) >= 2:
            subagent_type = args[1]
        if runtime is None and len(args) >= 3:
            runtime = args[2]
        return subagent_type if isinstance(subagent_type, str) else None, runtime

    if callable(original_sync):
        @wraps(original_sync)
        def wrapped_sync(*args: Any, **kwargs: Any) -> Any:
            subagent_type, runtime = _extract_call_details(args, kwargs)
            token = _set_pending_context(subagent_type=subagent_type, runtime=runtime)
            try:
                return original_sync(*args, **kwargs)
            finally:
                if token is not None:
                    _PENDING_CHILD_AGENT.reset(token)

        tool.func = wrapped_sync

    if callable(original_async):
        @wraps(original_async)
        async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
            subagent_type, runtime = _extract_call_details(args, kwargs)
            token = _set_pending_context(subagent_type=subagent_type, runtime=runtime)
            try:
                return await original_async(*args, **kwargs)
            finally:
                if token is not None:
                    _PENDING_CHILD_AGENT.reset(token)

        tool.coroutine = wrapped_async


def _copy_retry_policy(retry_policy: Any, max_attempts: int) -> Any:
    if retry_policy is not None or RetryPolicy is None:
        return retry_policy
    return RetryPolicy(max_attempts=max_attempts)


def _compile_builder(builder: Any, agent: Any) -> Any:
    return builder.compile(
        checkpointer=getattr(agent, "checkpointer", None),
        cache=getattr(agent, "cache", None),
        store=getattr(agent, "store", None),
        interrupt_before=getattr(agent, "interrupt_before_nodes", None),
        interrupt_after=getattr(agent, "interrupt_after_nodes", None),
        debug=bool(getattr(agent, "debug", False)),
        name=getattr(agent, "name", None),
    )


def _prepare_compiled_agent(
    agent: Any,
    *,
    trace_context: TraceContext,
    config: Mapping[str, Any] | None,
    model_max_retries: int,
    cache: dict[int, Any],
) -> Any:
    if StateGraph is None or StateNodeSpec is None or RetryPolicy is None:
        return agent
    if CompiledStateGraph is not None and not isinstance(agent, CompiledStateGraph):
        return agent

    builder = copy.deepcopy(agent.builder)
    model_spec = builder.nodes.get("model")
    if model_spec is not None:
        builder.nodes["model"] = StateNodeSpec(
            runnable=model_spec.runnable,
            metadata=model_spec.metadata,
            input_schema=model_spec.input_schema,
            retry_policy=_copy_retry_policy(model_spec.retry_policy, model_max_retries),
            cache_policy=model_spec.cache_policy,
            ends=model_spec.ends,
            defer=model_spec.defer,
        )

    tools_spec = builder.nodes.get("tools")
    if tools_spec is not None:
        tool_node = tools_spec.runnable
        tools_by_name = getattr(tool_node, "_tools_by_name", None)
        if isinstance(tools_by_name, dict):
            task_tool = tools_by_name.get("task")
            if task_tool is not None:
                for func in (getattr(task_tool, "func", None), getattr(task_tool, "coroutine", None)):
                    subagent_graphs = _extract_task_subagent_graphs(func)
                    if subagent_graphs is None:
                        continue
                    for name, subagent in list(subagent_graphs.items()):
                        subagent_graphs[name] = _auto_trace_agent_impl(
                            subagent,
                            trace_context=_trace_context_for_child_agents(trace_context),
                            config=config,
                            agent_name=name,
                            agent_type="subagent",
                            model_max_retries=model_max_retries,
                            cache=cache,
                        )
                _wrap_task_tool_for_pending_child_context(task_tool)

    return _compile_builder(builder, agent)


def _is_compiled_state_graph(value: Any) -> bool:
    if CompiledStateGraph is None:
        return False
    return isinstance(value, CompiledStateGraph)


@dataclass(slots=True)
class TracedAgentRunnable:
    runnable: Any
    trace_context: TraceContext
    agent_name: str
    agent_type: str | None = None
    config: Mapping[str, Any] | None = None

    def _build_effective_trace_context(self) -> tuple[TraceContext, dict[str, Any]]:
        parent_context = _CURRENT_AGENT_EXECUTION.get()
        pending_child = _PENDING_CHILD_AGENT.get()

        effective_name = self.agent_name
        if pending_child is not None and pending_child.agent_name:
            effective_name = pending_child.agent_name

        effective_type = (
            pending_child.agent_type
            if pending_child is not None
            else self.agent_type or ("subagent" if parent_context is not None else "agent")
        )
        parent_agent_name = parent_context.agent_name if parent_context is not None else None
        tool_call_id = (
            pending_child.tool_call_id
            if pending_child is not None
            else parent_context.tool_call_id if parent_context is not None else None
        )

        tags = dict(self.trace_context.tags)
        metadata = dict(self.trace_context.metadata)
        span_metadata = dict(self.trace_context.span_metadata)

        tags.setdefault("agent_name", effective_name)
        tags.setdefault("agent_type", effective_type)
        metadata.setdefault("agent_name", effective_name)
        metadata.setdefault("agent_type", effective_type)
        span_metadata.setdefault("agent_name", effective_name)
        span_metadata.setdefault("agent_type", effective_type)

        if parent_agent_name is not None:
            metadata.setdefault("parent_agent_name", parent_agent_name)
            span_metadata.setdefault("parent_agent_name", parent_agent_name)
        if tool_call_id is not None:
            metadata.setdefault("tool_call_id", tool_call_id)
            span_metadata.setdefault("tool_call_id", tool_call_id)

        return (
            TraceContext(
                tags=tags,
                metadata=metadata,
                span_metadata=span_metadata,
                run_tags=self.trace_context.run_tags,
                user_id=self.trace_context.user_id,
                session_id=self.trace_context.session_id,
                client_request_id=self.trace_context.client_request_id,
                mlflow_run_name=self.trace_context.mlflow_run_name,
                run_description=self.trace_context.run_description,
                ensure_run=self.trace_context.ensure_run,
                request_preview=self.trace_context.request_preview,
                response_preview=self.trace_context.response_preview,
                request_preview_builder=self.trace_context.request_preview_builder,
                response_preview_builder=self.trace_context.response_preview_builder,
                tags_builder=self.trace_context.tags_builder,
                metadata_builder=self.trace_context.metadata_builder,
                preview_limit=self.trace_context.preview_limit,
                trace_name=self.trace_context.trace_name or effective_name,
                capture_root_span_io=self.trace_context.capture_root_span_io,
            ),
            {
                "agent_name": effective_name,
                "agent_type": effective_type,
                "parent_agent_name": parent_agent_name,
                "tool_call_id": tool_call_id,
            },
        )

    def _build_config(
        self,
        agent_metadata: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = _merge_config(self.config, config)
        metadata = dict(merged.get("metadata", {}))
        metadata.update(
            {str(key): value for key, value in agent_metadata.items() if value is not None}
        )
        if metadata:
            merged["metadata"] = metadata
        merged.setdefault("run_name", self.agent_name)
        return merged

    def _enter_agent_execution(
        self,
        agent_metadata: Mapping[str, Any],
    ) -> tuple[contextvars.Token[AgentExecutionContext | None], contextvars.Token[PendingChildAgentContext | None] | None]:
        pending_token: contextvars.Token[PendingChildAgentContext | None] | None = None
        if _PENDING_CHILD_AGENT.get() is not None:
            pending_token = _PENDING_CHILD_AGENT.set(None)

        execution_token = _CURRENT_AGENT_EXECUTION.set(
            AgentExecutionContext(
                agent_name=str(agent_metadata["agent_name"]),
                agent_type=str(agent_metadata["agent_type"]),
                tool_call_id=(
                    str(agent_metadata["tool_call_id"])
                    if agent_metadata.get("tool_call_id") is not None
                    else None
                ),
            )
        )
        return execution_token, pending_token

    def invoke(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        effective_trace_context, agent_metadata = self._build_effective_trace_context()
        handle = open_traced_span(effective_trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _AGENT_SPAN_TYPE)
        _set_span_attributes(handle.span, agent_metadata)
        span_token = _push_current_agent_span(handle.span)
        binding = _CURRENT_AGENT_SPAN_STACK.get()[-1]
        execution_token, pending_token = self._enter_agent_execution(agent_metadata)
        try:
            result = self.runnable.invoke(
                inputs,
                config=self._build_config(agent_metadata, config),
                **kwargs,
            )
            handle.response = _build_agent_span_output(result, binding)
            return result
        except BaseException as exc:
            handle.error = exc
            raise
        finally:
            try:
                try:
                    _CURRENT_AGENT_EXECUTION.reset(execution_token)
                    if pending_token is not None:
                        _PENDING_CHILD_AGENT.reset(pending_token)
                finally:
                    _reset_current_agent_span(span_token)
            finally:
                close_traced_span(handle)

    async def ainvoke(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        effective_trace_context, agent_metadata = self._build_effective_trace_context()
        handle = open_traced_span(effective_trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _AGENT_SPAN_TYPE)
        _set_span_attributes(handle.span, agent_metadata)
        span_token = _push_current_agent_span(handle.span)
        binding = _CURRENT_AGENT_SPAN_STACK.get()[-1]
        execution_token, pending_token = self._enter_agent_execution(agent_metadata)
        try:
            result = await self.runnable.ainvoke(
                inputs,
                config=self._build_config(agent_metadata, config),
                **kwargs,
            )
            handle.response = _build_agent_span_output(result, binding)
            return result
        except BaseException as exc:
            handle.error = exc
            raise
        finally:
            try:
                try:
                    _CURRENT_AGENT_EXECUTION.reset(execution_token)
                    if pending_token is not None:
                        _PENDING_CHILD_AGENT.reset(pending_token)
                finally:
                    _reset_current_agent_span(span_token)
            finally:
                close_traced_span(handle)

    def stream(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        effective_trace_context, agent_metadata = self._build_effective_trace_context()
        handle = open_traced_span(effective_trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _AGENT_SPAN_TYPE)
        _set_span_attributes(handle.span, agent_metadata)
        span_token = _push_current_agent_span(handle.span)
        binding = _CURRENT_AGENT_SPAN_STACK.get()[-1]
        execution_token, pending_token = self._enter_agent_execution(agent_metadata)

        try:
            iterator = self.runnable.stream(
                inputs,
                config=self._build_config(agent_metadata, config),
                **kwargs,
            )
        except BaseException as exc:
            handle.error = exc
            try:
                try:
                    _CURRENT_AGENT_EXECUTION.reset(execution_token)
                    if pending_token is not None:
                        _PENDING_CHILD_AGENT.reset(pending_token)
                finally:
                    _reset_current_agent_span(span_token)
            finally:
                close_traced_span(handle)
            raise

        def _stream() -> Iterator[Any]:
            last_chunk: Any = None
            try:
                for chunk in iterator:
                    last_chunk = chunk
                    yield chunk
                handle.response = _build_agent_span_output(last_chunk, binding)
            except BaseException as exc:
                handle.error = exc
                raise
            finally:
                try:
                    try:
                        _CURRENT_AGENT_EXECUTION.reset(execution_token)
                        if pending_token is not None:
                            _PENDING_CHILD_AGENT.reset(pending_token)
                    finally:
                        _reset_current_agent_span(span_token)
                finally:
                    close_traced_span(handle)

        return _stream()

    async def astream(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        effective_trace_context, agent_metadata = self._build_effective_trace_context()
        handle = open_traced_span(effective_trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _AGENT_SPAN_TYPE)
        _set_span_attributes(handle.span, agent_metadata)
        span_token = _push_current_agent_span(handle.span)
        binding = _CURRENT_AGENT_SPAN_STACK.get()[-1]
        execution_token, pending_token = self._enter_agent_execution(agent_metadata)

        try:
            iterator = self.runnable.astream(
                inputs,
                config=self._build_config(agent_metadata, config),
                **kwargs,
            )
        except BaseException as exc:
            handle.error = exc
            try:
                try:
                    _CURRENT_AGENT_EXECUTION.reset(execution_token)
                    if pending_token is not None:
                        _PENDING_CHILD_AGENT.reset(pending_token)
                finally:
                    _reset_current_agent_span(span_token)
            finally:
                close_traced_span(handle)
            raise

        try:
            last_chunk: Any = None
            async for chunk in iterator:
                last_chunk = chunk
                yield chunk
            handle.response = _build_agent_span_output(last_chunk, binding)
        except BaseException as exc:
            handle.error = exc
            raise
        finally:
            try:
                try:
                    _CURRENT_AGENT_EXECUTION.reset(execution_token)
                    if pending_token is not None:
                        _PENDING_CHILD_AGENT.reset(pending_token)
                finally:
                    _reset_current_agent_span(span_token)
            finally:
                close_traced_span(handle)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runnable, name)


def _auto_trace_agent_impl(
    agent: Any,
    *,
    trace_context: TraceContext,
    config: Mapping[str, Any] | None,
    agent_name: str | None,
    agent_type: str | None,
    model_max_retries: int,
    cache: dict[int, Any],
) -> TracedAgentRunnable:
    cached = cache.get(id(agent))
    if cached is not None:
        return cached

    prepared_agent = (
        _prepare_compiled_agent(
            agent,
            trace_context=trace_context,
            config=config,
            model_max_retries=model_max_retries,
            cache=cache,
        )
        if _is_compiled_state_graph(agent)
        else agent
    )
    wrapped = TracedAgentRunnable(
        runnable=prepared_agent,
        trace_context=trace_context,
        agent_name=_resolve_agent_name(prepared_agent, trace_context, explicit_name=agent_name),
        agent_type=agent_type,
        config=config,
    )
    cache[id(agent)] = wrapped
    return wrapped


def auto_trace_agent(
    agent: Any,
    trace_context: TraceContextLike = None,
    /,
    *,
    config: Mapping[str, Any] | None = None,
    agent_name: str | None = None,
    agent_type: str | None = None,
    model_max_retries: int = 3,
    **trace_context_kwargs: Any,
) -> TracedAgentRunnable:
    resolved_trace_context = _coerce_trace_context(trace_context, **trace_context_kwargs)
    return _auto_trace_agent_impl(
        agent,
        trace_context=resolved_trace_context,
        config=config,
        agent_name=agent_name,
        agent_type=agent_type,
        model_max_retries=model_max_retries,
        cache={},
    )

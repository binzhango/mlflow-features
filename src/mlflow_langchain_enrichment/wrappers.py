from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Mapping

from pydantic import ConfigDict, Field

from .agent import has_current_agent_span, record_current_agent_model_observation
from .auto import (
    accumulate_current_trace_token_usage,
    close_traced_span,
    open_traced_span,
    record_current_trace_token_usage,
)
from .enrichment import TraceContext, build_runnable_config

try:
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:  # pragma: no cover - optional dependency at import time
    BaseChatModel = None  # type: ignore[assignment,misc]


TraceContextLike = TraceContext | Mapping[str, Any] | None
_CHAT_TOKEN_USAGE_ATTRIBUTE_KEY = "mlflow.chat.tokenUsage"
_TOKEN_USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")
_CHAT_MODEL_SPAN_TYPE = "CHAT_MODEL"
_TOOL_SPAN_TYPE = "TOOL"


def _set_span_type(span: Any, span_type: str) -> None:
    if hasattr(span, "set_span_type"):
        span.set_span_type(span_type)


def _resolve_model_span_type(response: Any) -> str:
    if isinstance(response, Mapping):
        if response.get("tool_calls"):
            return _TOOL_SPAN_TYPE
        return _CHAT_MODEL_SPAN_TYPE

    tool_calls = getattr(response, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        return _TOOL_SPAN_TYPE

    message = getattr(response, "message", None)
    if message is not None and message is not response:
        return _resolve_model_span_type(message)

    generations = getattr(response, "generations", None)
    if isinstance(generations, list) and generations:
        for item in generations:
            span_type = _resolve_model_span_type(item)
            if span_type == _TOOL_SPAN_TYPE:
                return span_type

    return _CHAT_MODEL_SPAN_TYPE


def coerce_trace_context(trace_context: TraceContextLike) -> TraceContext | None:
    if trace_context is None:
        return None
    if isinstance(trace_context, TraceContext):
        return trace_context
    if isinstance(trace_context, Mapping):
        return TraceContext(**dict(trace_context))
    raise TypeError(
        "Trace context must be a TraceContext, a mapping of TraceContext fields, or None."
    )


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

        if key == "callbacks" and isinstance(merged.get(key), list) and isinstance(value, list):
            merged[key] = [*merged[key], *value]
            continue

        merged[key] = value

    return merged


def _is_base_chat_model_like(value: Any) -> bool:
    if BaseChatModel is None:
        return False
    if isinstance(value, BaseChatModel):
        return True
    bound = getattr(value, "bound", None)
    return isinstance(bound, BaseChatModel)


def _coerce_usage_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_from_mapping(mapping: Mapping[str, Any] | None) -> dict[str, int]:
    if not mapping:
        return {}

    input_tokens = _coerce_usage_int(
        mapping.get("input_tokens", mapping.get("prompt_eval_count"))
    )
    output_tokens = _coerce_usage_int(
        mapping.get("output_tokens", mapping.get("eval_count"))
    )
    total_tokens = _coerce_usage_int(mapping.get("total_tokens"))
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
        usage = _usage_from_mapping(response.get("usage_metadata"))
        response_usage = _usage_from_mapping(response.get("response_metadata"))
        direct_usage = _usage_from_mapping(response)
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

    usage = _usage_from_mapping(getattr(response, "usage_metadata", None))
    response_usage = _usage_from_mapping(getattr(response, "response_metadata", None))
    merged = {**response_usage, **usage}
    return {key: merged[key] for key in _TOKEN_USAGE_KEYS if key in merged}


def _trace_context_with_token_usage(
    trace_context: TraceContext,
    response: Any,
) -> TraceContext:
    # Reserve trace-level token metadata for root trace close only, using the
    # accumulator in auto.py. This keeps mlflow.trace.tokenUsage from being
    # overwritten mid-run by individual model calls.
    return trace_context


def _set_span_token_usage(span: Any, response: Any) -> None:
    usage = _extract_token_usage(response)
    if not usage:
        return
    if hasattr(span, "set_attribute"):
        span.set_attribute(_CHAT_TOKEN_USAGE_ATTRIBUTE_KEY, usage)
        return
    if hasattr(span, "set_attributes"):
        span.set_attributes({_CHAT_TOKEN_USAGE_ATTRIBUTE_KEY: usage})


def _resolve_model_name_for_ledger(
    runnable: Any,
    trace_context: TraceContext,
    response: Any,
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


def _record_trace_token_usage_for_result(
    runnable: Any,
    trace_context: TraceContext,
    response: Any,
) -> None:
    usage = _extract_token_usage(response)
    if not usage:
        return
    if not has_current_agent_span():
        accumulate_current_trace_token_usage(usage)
    record_current_trace_token_usage(
        usage,
        span_name=trace_context.trace_name,
        model_name=_resolve_model_name_for_ledger(runnable, trace_context, response),
    )


class _StreamAccumulator:
    def __init__(self) -> None:
        self._aggregate: Any = None
        self._chunks: list[Any] = []
        self._can_concatenate = True

    def add(self, chunk: Any) -> None:
        self._chunks.append(chunk)
        if not self._can_concatenate:
            return
        if self._aggregate is None:
            self._aggregate = chunk
            return
        try:
            self._aggregate = self._aggregate + chunk
        except Exception:
            self._can_concatenate = False

    def result(self) -> Any:
        if self._can_concatenate and self._aggregate is not None:
            return self._aggregate
        if not self._chunks:
            return None
        if len(self._chunks) == 1:
            return self._chunks[0]
        return list(self._chunks)


@dataclass(slots=True)
class PlainTracedRunnable:
    runnable: Any
    trace_context: TraceContext
    config: Mapping[str, Any] | None = None

    def _build_config(self, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return build_runnable_config(
            self.trace_context,
            _merge_config(self.config, config),
            include_callback=False,
        )

    def _apply_dynamic_trace_attributes(
        self,
        *,
        inputs: Any,
        response: Any = None,
        error: BaseException | None = None,
    ) -> TraceContext:
        tags = dict(self.trace_context.tags)
        metadata = dict(self.trace_context.metadata)

        if self.trace_context.tags_builder is not None:
            built_tags = self.trace_context.tags_builder(self.runnable, inputs, response, error) or {}
            tags.update(built_tags)

        if self.trace_context.metadata_builder is not None:
            built_metadata = (
                self.trace_context.metadata_builder(self.runnable, inputs, response, error) or {}
            )
            metadata.update(built_metadata)

        if tags == dict(self.trace_context.tags) and metadata == dict(self.trace_context.metadata):
            return self.trace_context

        return TraceContext(
            tags=tags,
            metadata=metadata,
            span_metadata=self.trace_context.span_metadata,
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
            trace_name=self.trace_context.trace_name,
            capture_root_span_io=self.trace_context.capture_root_span_io,
        )

    def invoke(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        try:
            result = self.runnable.invoke(inputs, config=self._build_config(config), **kwargs)
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            close_traced_span(handle)

    async def ainvoke(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        try:
            result = await self.runnable.ainvoke(inputs, config=self._build_config(config), **kwargs)
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            close_traced_span(handle)

    def stream(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        accumulator = _StreamAccumulator()

        try:
            iterator = self.runnable.stream(inputs, config=self._build_config(config), **kwargs)
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            close_traced_span(handle)
            raise

        def _stream() -> Iterator[Any]:
            try:
                for chunk in iterator:
                    accumulator.add(chunk)
                    yield chunk
                result = accumulator.result()
                handle.trace_context = self._apply_dynamic_trace_attributes(
                    inputs=inputs,
                    response=result,
                )
                handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
                _set_span_token_usage(handle.span, result)
                _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
                record_current_agent_model_observation(
                    trace_context=handle.trace_context,
                    runnable=self.runnable,
                    response=result,
                )
                handle.response = result
            except BaseException as exc:
                handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
                handle.error = exc
                raise
            finally:
                if handle.error is None and handle.response is None:
                    result = accumulator.result()
                    handle.trace_context = self._apply_dynamic_trace_attributes(
                        inputs=inputs,
                        response=result,
                    )
                    handle.trace_context = _trace_context_with_token_usage(
                        handle.trace_context,
                        result,
                    )
                    _set_span_token_usage(handle.span, result)
                    _record_trace_token_usage_for_result(
                        self.runnable,
                        handle.trace_context,
                        result,
                    )
                    record_current_agent_model_observation(
                        trace_context=handle.trace_context,
                        runnable=self.runnable,
                        response=result,
                    )
                    handle.response = result
                close_traced_span(handle)

        return _stream()

    async def astream(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        accumulator = _StreamAccumulator()

        try:
            iterator = self.runnable.astream(inputs, config=self._build_config(config), **kwargs)
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            close_traced_span(handle)
            raise

        try:
            async for chunk in iterator:
                accumulator.add(chunk)
                yield chunk
            result = accumulator.result()
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            if handle.error is None and handle.response is None:
                result = accumulator.result()
                handle.trace_context = self._apply_dynamic_trace_attributes(
                    inputs=inputs,
                    response=result,
                )
                handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
                _set_span_token_usage(handle.span, result)
                _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
                record_current_agent_model_observation(
                    trace_context=handle.trace_context,
                    runnable=self.runnable,
                    response=result,
                )
                handle.response = result
            close_traced_span(handle)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runnable, name)


class TracedChatModel(BaseChatModel):  # type: ignore[misc,valid-type]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runnable: Any = Field(exclude=True)
    trace_context: TraceContext = Field(exclude=True)
    base_config: Mapping[str, Any] | None = Field(default=None, exclude=True)

    def __init__(
        self,
        runnable: Any,
        trace_context: TraceContext,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(runnable=runnable, trace_context=trace_context, base_config=config)

    def _build_config(self, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return build_runnable_config(
            self.trace_context,
            _merge_config(self.base_config, config),
            include_callback=False,
        )

    def _apply_dynamic_trace_attributes(
        self,
        *,
        inputs: Any,
        response: Any = None,
        error: BaseException | None = None,
    ) -> TraceContext:
        tags = dict(self.trace_context.tags)
        metadata = dict(self.trace_context.metadata)

        if self.trace_context.tags_builder is not None:
            built_tags = self.trace_context.tags_builder(
                self.runnable,
                inputs,
                response,
                error,
            ) or {}
            tags.update(built_tags)

        if self.trace_context.metadata_builder is not None:
            built_metadata = self.trace_context.metadata_builder(
                self.runnable,
                inputs,
                response,
                error,
            ) or {}
            metadata.update(built_metadata)

        if tags == dict(self.trace_context.tags) and metadata == dict(self.trace_context.metadata):
            return self.trace_context

        return TraceContext(
            tags=tags,
            metadata=metadata,
            span_metadata=self.trace_context.span_metadata,
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
            trace_name=self.trace_context.trace_name,
            capture_root_span_io=self.trace_context.capture_root_span_io,
        )

    @property
    def _llm_type(self) -> str:
        llm_type = getattr(self.runnable, "_llm_type", None)
        return llm_type if isinstance(llm_type, str) else type(self.runnable).__name__.lower()

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        handle = open_traced_span(self.trace_context)
        handle.request = messages
        _set_span_type(handle.span, _CHAT_MODEL_SPAN_TYPE)
        try:
            result = self.runnable._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            _set_span_type(handle.span, _resolve_model_span_type(result))
            handle.trace_context = self._apply_dynamic_trace_attributes(
                inputs=messages,
                response=result,
            )
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=messages, error=exc)
            handle.error = exc
            raise
        finally:
            close_traced_span(handle)

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        handle = open_traced_span(self.trace_context)
        handle.request = messages
        _set_span_type(handle.span, _CHAT_MODEL_SPAN_TYPE)
        try:
            result = await self.runnable._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            _set_span_type(handle.span, _resolve_model_span_type(result))
            handle.trace_context = self._apply_dynamic_trace_attributes(
                inputs=messages,
                response=result,
            )
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=messages, error=exc)
            handle.error = exc
            raise
        finally:
            close_traced_span(handle)

    def invoke(
        self,
        inputs: Any,
        config: Mapping[str, Any] | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _CHAT_MODEL_SPAN_TYPE)
        try:
            result = self.runnable.invoke(
                inputs,
                config=self._build_config(config),
                stop=stop,
                **kwargs,
            )
            _set_span_type(handle.span, _resolve_model_span_type(result))
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            close_traced_span(handle)

    async def ainvoke(
        self,
        inputs: Any,
        config: Mapping[str, Any] | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _CHAT_MODEL_SPAN_TYPE)
        try:
            result = await self.runnable.ainvoke(
                inputs,
                config=self._build_config(config),
                stop=stop,
                **kwargs,
            )
            _set_span_type(handle.span, _resolve_model_span_type(result))
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            close_traced_span(handle)

    def stream(
        self,
        inputs: Any,
        config: Mapping[str, Any] | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _CHAT_MODEL_SPAN_TYPE)
        accumulator = _StreamAccumulator()

        try:
            iterator = self.runnable.stream(
                inputs,
                config=self._build_config(config),
                stop=stop,
                **kwargs,
            )
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            close_traced_span(handle)
            raise

        def _stream() -> Iterator[Any]:
            try:
                for chunk in iterator:
                    accumulator.add(chunk)
                    yield chunk
                result = accumulator.result()
                _set_span_type(handle.span, _resolve_model_span_type(result))
                handle.trace_context = self._apply_dynamic_trace_attributes(
                    inputs=inputs,
                    response=result,
                )
                handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
                _set_span_token_usage(handle.span, result)
                _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
                record_current_agent_model_observation(
                    trace_context=handle.trace_context,
                    runnable=self.runnable,
                    response=result,
                )
                handle.response = result
            except BaseException as exc:
                handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
                handle.error = exc
                raise
            finally:
                if handle.error is None and handle.response is None:
                    result = accumulator.result()
                    _set_span_type(handle.span, _resolve_model_span_type(result))
                    handle.trace_context = self._apply_dynamic_trace_attributes(
                        inputs=inputs,
                        response=result,
                    )
                    handle.trace_context = _trace_context_with_token_usage(
                        handle.trace_context,
                        result,
                    )
                    _set_span_token_usage(handle.span, result)
                    _record_trace_token_usage_for_result(
                        self.runnable,
                        handle.trace_context,
                        result,
                    )
                    record_current_agent_model_observation(
                        trace_context=handle.trace_context,
                        runnable=self.runnable,
                        response=result,
                    )
                    handle.response = result
                close_traced_span(handle)

        return _stream()

    async def astream(
        self,
        inputs: Any,
        config: Mapping[str, Any] | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        handle = open_traced_span(self.trace_context)
        handle.request = inputs
        _set_span_type(handle.span, _CHAT_MODEL_SPAN_TYPE)
        accumulator = _StreamAccumulator()

        try:
            iterator = self.runnable.astream(
                inputs,
                config=self._build_config(config),
                stop=stop,
                **kwargs,
            )
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            close_traced_span(handle)
            raise

        try:
            async for chunk in iterator:
                accumulator.add(chunk)
                yield chunk
            result = accumulator.result()
            _set_span_type(handle.span, _resolve_model_span_type(result))
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
            _set_span_token_usage(handle.span, result)
            _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
            record_current_agent_model_observation(
                trace_context=handle.trace_context,
                runnable=self.runnable,
                response=result,
            )
            handle.response = result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            if handle.error is None and handle.response is None:
                result = accumulator.result()
                _set_span_type(handle.span, _resolve_model_span_type(result))
                handle.trace_context = self._apply_dynamic_trace_attributes(
                    inputs=inputs,
                    response=result,
                )
                handle.trace_context = _trace_context_with_token_usage(handle.trace_context, result)
                _set_span_token_usage(handle.span, result)
                _record_trace_token_usage_for_result(self.runnable, handle.trace_context, result)
                record_current_agent_model_observation(
                    trace_context=handle.trace_context,
                    runnable=self.runnable,
                    response=result,
                )
                handle.response = result
            close_traced_span(handle)

    def bind_tools(self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any) -> Any:
        bound = self.runnable.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        if _is_base_chat_model_like(bound):
            return wrap_llm(bound, self.trace_context, config=self.base_config)
        return wrap_runnable(bound, self.trace_context, config=self.base_config)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        structured = self.runnable.with_structured_output(
            schema,
            include_raw=include_raw,
            **kwargs,
        )
        return wrap_runnable(structured, self.trace_context, config=self.base_config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runnable, name)


def wrap_runnable(
    runnable: Any,
    trace_context: TraceContextLike = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> PlainTracedRunnable:
    resolved_trace_context = coerce_trace_context(trace_context) or TraceContext()
    return PlainTracedRunnable(runnable=runnable, trace_context=resolved_trace_context, config=config)


def wrap_chain(
    chain: Any,
    trace_context: TraceContextLike = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> PlainTracedRunnable:
    return wrap_runnable(chain, trace_context, config=config)


def wrap_llm(
    llm: Any,
    trace_context: TraceContextLike = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> Any:
    resolved_trace_context = coerce_trace_context(trace_context) or TraceContext()
    if _is_base_chat_model_like(llm):
        return TracedChatModel(llm, resolved_trace_context, config=config)
    return PlainTracedRunnable(runnable=llm, trace_context=resolved_trace_context, config=config)

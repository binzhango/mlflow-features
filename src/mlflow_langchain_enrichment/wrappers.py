from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Mapping

from .auto import close_root_trace, open_root_trace
from .enrichment import TraceContext, build_runnable_config


TraceContextLike = TraceContext | Mapping[str, Any] | None


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
        handle = open_root_trace(self.trace_context)
        handle.request = inputs
        try:
            result = self.runnable.invoke(inputs, config=self._build_config(config), **kwargs)
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            close_root_trace(handle)

    async def ainvoke(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        handle = open_root_trace(self.trace_context)
        handle.request = inputs
        try:
            result = await self.runnable.ainvoke(inputs, config=self._build_config(config), **kwargs)
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
            handle.response = result
            return result
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            raise
        finally:
            close_root_trace(handle)

    def stream(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        handle = open_root_trace(self.trace_context)
        handle.request = inputs
        accumulator = _StreamAccumulator()

        try:
            iterator = self.runnable.stream(inputs, config=self._build_config(config), **kwargs)
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            close_root_trace(handle)
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
                    handle.response = result
                close_root_trace(handle)

        return _stream()

    async def astream(
        self,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        handle = open_root_trace(self.trace_context)
        handle.request = inputs
        accumulator = _StreamAccumulator()

        try:
            iterator = self.runnable.astream(inputs, config=self._build_config(config), **kwargs)
        except BaseException as exc:
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, error=exc)
            handle.error = exc
            close_root_trace(handle)
            raise

        try:
            async for chunk in iterator:
                accumulator.add(chunk)
                yield chunk
            result = accumulator.result()
            handle.trace_context = self._apply_dynamic_trace_attributes(inputs=inputs, response=result)
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
                handle.response = result
            close_root_trace(handle)

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
) -> PlainTracedRunnable:
    return wrap_runnable(llm, trace_context, config=config)

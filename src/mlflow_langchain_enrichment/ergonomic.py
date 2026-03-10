from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Mapping

from .auto import RootTraceHandle, close_root_trace, open_root_trace
from .enrichment import TraceContext
from .wrappers import TraceContextLike, coerce_trace_context, wrap_chain, wrap_llm, wrap_runnable

RequestResolver = Callable[..., Any]

_REQUEST_ARG_NAMES = ("input", "inputs", "payload", "messages", "question", "request")


def _resolve_trace_context(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> TraceContext:
    if trace_context is not None and trace_context_kwargs:
        raise ValueError("Pass either a TraceContext object or keyword fields, not both.")

    resolved = coerce_trace_context(trace_context)
    if resolved is None:
        resolved = TraceContext(**trace_context_kwargs)
    return resolved


def _looks_like_runnable(value: Any) -> bool:
    return any(hasattr(value, name) for name in ("invoke", "ainvoke", "stream", "astream"))


def _resolve_request_from_call(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
    except TypeError:
        bound = None

    if bound is not None:
        for key in _REQUEST_ARG_NAMES:
            if key in bound.arguments:
                return bound.arguments[key]

        values = [
            value
            for name, value in bound.arguments.items()
            if name not in ("self", "cls")
        ]
        if len(values) >= 2 and _looks_like_runnable(values[0]):
            return values[1]
        if values:
            return values[0]

    for key in _REQUEST_ARG_NAMES:
        if key in kwargs:
            return kwargs[key]

    if len(args) >= 2 and _looks_like_runnable(args[0]):
        return args[1]
    if args:
        return args[0]
    return None


@dataclass(slots=True)
class TraceSession:
    trace_context: TraceContext
    _handle: RootTraceHandle | None = None

    def _sync_current_trace(self) -> None:
        if self._handle is None:
            return

        import mlflow

        mlflow.update_current_trace(
            tags=self.trace_context.trace_tags() or None,
            metadata=self.trace_context.trace_metadata() or None,
            client_request_id=self.trace_context.client_request_id,
        )

    def set_request(self, request: Any) -> Any:
        if self._handle is None:
            raise RuntimeError("Trace session has not been entered.")
        self._handle.request = request
        return request

    def set_response(self, response: Any) -> Any:
        if self._handle is None:
            raise RuntimeError("Trace session has not been entered.")
        self._handle.response = response
        return response

    def set_error(self, error: BaseException) -> BaseException:
        if self._handle is None:
            raise RuntimeError("Trace session has not been entered.")
        self._handle.error = error
        return error

    def add_tags(self, tags: Mapping[str, Any]) -> dict[str, Any]:
        merged = {**self.trace_context.tags, **dict(tags)}
        self.trace_context.tags = merged
        self._sync_current_trace()
        return merged

    def add_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        merged = {**self.trace_context.metadata, **dict(metadata)}
        self.trace_context.metadata = merged
        self._sync_current_trace()
        return merged

    def update_trace(
        self,
        *,
        tags: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if tags:
            self.add_tags(tags)
        if metadata:
            self.add_metadata(metadata)

    def __enter__(self) -> "TraceSession":
        self._handle = open_root_trace(self.trace_context)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        if self._handle is None:
            return False
        if exc is not None:
            self._handle.error = exc
        close_root_trace(self._handle)
        self._handle = None
        return False

    async def __aenter__(self) -> "TraceSession":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        return self.__exit__(exc_type, exc, tb)


def auto_trace_runnable(
    runnable: Any,
    trace_context: TraceContextLike = None,
    /,
    *,
    config: Mapping[str, Any] | None = None,
    **trace_context_kwargs: Any,
) -> Any:
    return wrap_runnable(
        runnable,
        _resolve_trace_context(trace_context, **trace_context_kwargs),
        config=config,
    )


def auto_trace_chain(
    chain: Any,
    trace_context: TraceContextLike = None,
    /,
    *,
    config: Mapping[str, Any] | None = None,
    **trace_context_kwargs: Any,
) -> Any:
    return wrap_chain(
        chain,
        _resolve_trace_context(trace_context, **trace_context_kwargs),
        config=config,
    )


def auto_trace_llm(
    llm: Any,
    trace_context: TraceContextLike = None,
    /,
    *,
    config: Mapping[str, Any] | None = None,
    **trace_context_kwargs: Any,
) -> Any:
    return wrap_llm(
        llm,
        _resolve_trace_context(trace_context, **trace_context_kwargs),
        config=config,
    )


def trace_llm(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> TraceSession:
    return TraceSession(_resolve_trace_context(trace_context, **trace_context_kwargs))


def trace_llm_call(
    trace_context: TraceContextLike = None,
    /,
    *,
    request_resolver: RequestResolver | None = None,
    **trace_context_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    resolved_trace_context = _resolve_trace_context(trace_context, **trace_context_kwargs)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                resolved_request = (
                    request_resolver(*args, **kwargs)
                    if request_resolver is not None
                    else _resolve_request_from_call(func, args, kwargs)
                )
                async with trace_llm(resolved_trace_context) as trace:
                    if resolved_request is not None:
                        trace.set_request(resolved_request)
                    result = await func(*args, **kwargs)
                    trace.set_response(result)
                    return result

            return async_wrapped

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            resolved_request = (
                request_resolver(*args, **kwargs)
                if request_resolver is not None
                else _resolve_request_from_call(func, args, kwargs)
            )
            with trace_llm(resolved_trace_context) as trace:
                if resolved_request is not None:
                    trace.set_request(resolved_request)
                result = func(*args, **kwargs)
                trace.set_response(result)
                return result

        return wrapped

    return decorator

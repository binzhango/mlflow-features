from __future__ import annotations

import contextlib
import contextvars
import threading
import warnings
from dataclasses import dataclass
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager

from .enrichment import (
    TraceContext,
    TraceEnrichmentCallback,
    to_root_span_inputs,
    to_root_span_outputs,
    to_span_io,
    update_trace_from_context,
)


TraceContextLike = TraceContext | Mapping[str, Any] | None
TraceContextProvider = Callable[[], TraceContextLike]

_CURRENT_TRACE_CONTEXT: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "mlflow_langchain_enrichment_trace_context",
    default=None,
)
_CONFIGURE_LOCK = threading.Lock()
_PATCHED = False
_TRACE_CONTEXT_PROVIDER: TraceContextProvider | None = None
_ORIGINAL_CALLBACK_CONFIGURE: Callable[..., CallbackManager] | None = None
_ORIGINAL_ASYNC_CALLBACK_CONFIGURE: Callable[..., AsyncCallbackManager] | None = None


@dataclass(slots=True)
class TraceContextHandle:
    trace_context: TraceContext
    token: contextvars.Token[TraceContext | None]
    exit_stack: contextlib.ExitStack


@dataclass(slots=True)
class RootTraceHandle:
    trace_context: TraceContext
    trace_context_handle: TraceContextHandle
    span: Any
    exit_stack: contextlib.ExitStack
    request: Any = None
    response: Any = None
    error: BaseException | None = None


def _coerce_trace_context(trace_context: TraceContextLike) -> TraceContext | None:
    if trace_context is None:
        return None
    if isinstance(trace_context, TraceContext):
        return trace_context
    if isinstance(trace_context, Mapping):
        return TraceContext(**dict(trace_context))
    raise TypeError(
        "Trace context must be a TraceContext, a mapping of TraceContext fields, or None."
    )


def get_current_trace_context() -> TraceContext | None:
    return _CURRENT_TRACE_CONTEXT.get()


def set_current_trace_context(
    trace_context: TraceContextLike,
) -> contextvars.Token[TraceContext | None]:
    return _CURRENT_TRACE_CONTEXT.set(_coerce_trace_context(trace_context))


def reset_current_trace_context(token: contextvars.Token[TraceContext | None]) -> None:
    _CURRENT_TRACE_CONTEXT.reset(token)


def _build_trace_context(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> TraceContext:
    if trace_context is not None and trace_context_kwargs:
        raise ValueError("Pass either a TraceContext object or keyword fields, not both.")

    resolved = _coerce_trace_context(trace_context)
    if resolved is None:
        resolved = TraceContext(**trace_context_kwargs)
    return resolved


def open_trace_context(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> TraceContextHandle:
    resolved = _build_trace_context(trace_context, **trace_context_kwargs)
    token = set_current_trace_context(resolved)
    exit_stack = contextlib.ExitStack()

    try:
        if (
            resolved.ensure_run
            or resolved.mlflow_run_name is not None
            or resolved.run_description is not None
            or resolved.run_tags
        ):
            import mlflow

            if mlflow.active_run() is None:
                exit_stack.enter_context(
                    mlflow.start_run(
                        run_name=resolved.mlflow_run_name or resolved.trace_name,
                        tags=dict(resolved.run_tags) or None,
                        description=resolved.run_description,
                    )
                )

        return TraceContextHandle(
            trace_context=resolved,
            token=token,
            exit_stack=exit_stack,
        )
    except Exception:
        exit_stack.close()
        reset_current_trace_context(token)
        raise


def close_trace_context(handle: TraceContextHandle) -> None:
    try:
        handle.exit_stack.close()
    finally:
        reset_current_trace_context(handle.token)


def open_root_trace(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> RootTraceHandle:
    resolved = _build_trace_context(trace_context, **trace_context_kwargs)
    trace_context_handle = open_trace_context(resolved)
    exit_stack = contextlib.ExitStack()

    try:
        import mlflow

        span = exit_stack.enter_context(
            mlflow.start_span(name=resolved.trace_name or "request")
        )
        update_trace_from_context(resolved)
        return RootTraceHandle(
            trace_context=resolved,
            trace_context_handle=trace_context_handle,
            span=span,
            exit_stack=exit_stack,
        )
    except Exception:
        exit_stack.close()
        close_trace_context(trace_context_handle)
        raise


def close_root_trace(
    handle: RootTraceHandle,
    *,
    request: Any = None,
    response: Any = None,
    error: BaseException | None = None,
) -> None:
    try:
        final_request = request if request is not None else handle.request
        final_error = error if error is not None else handle.error
        final_response = response if response is not None else handle.response

        if final_request is not None and handle.trace_context.capture_root_span_io:
            handle.span.set_inputs(to_root_span_inputs(final_request))

        if final_error is not None:
            if handle.trace_context.capture_root_span_io:
                handle.span.set_outputs(
                    {"error": f"{type(final_error).__name__}: {final_error}"}
                )
            update_trace_from_context(
                handle.trace_context,
                request=final_request,
                response=f"{type(final_error).__name__}: {final_error}",
                state="ERROR",
            )
        elif final_response is not None:
            if handle.trace_context.capture_root_span_io:
                handle.span.set_outputs(to_root_span_outputs(final_response))
            update_trace_from_context(
                handle.trace_context,
                request=final_request,
                response=final_response,
                state="OK",
            )
        elif final_request is not None:
            update_trace_from_context(
                handle.trace_context,
                request=final_request,
            )
    finally:
        try:
            handle.exit_stack.close()
        finally:
            close_trace_context(handle.trace_context_handle)


@contextlib.contextmanager
def using_trace_context(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> Iterator[TraceContext]:
    handle = open_trace_context(trace_context, **trace_context_kwargs)
    try:
        yield handle.trace_context
    finally:
        close_trace_context(handle)


@contextlib.contextmanager
def using_root_trace(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> Iterator[RootTraceHandle]:
    handle = open_root_trace(trace_context, **trace_context_kwargs)
    try:
        yield handle
    except BaseException as exc:
        handle.error = exc
        close_root_trace(handle)
        raise
    else:
        close_root_trace(handle)


# Backward-compatible aliases for older code paths.
ManualRootTraceHandle = RootTraceHandle
open_manual_root_trace = open_root_trace
close_manual_root_trace = close_root_trace
using_manual_root_trace = using_root_trace


def _resolve_trace_context() -> TraceContext | None:
    current = get_current_trace_context()
    if current is not None:
        return current

    if _TRACE_CONTEXT_PROVIDER is None:
        return None

    try:
        return _coerce_trace_context(_TRACE_CONTEXT_PROVIDER())
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        warnings.warn(
            f"mlflow-langchain-enrichment context provider failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _has_trace_handler(callbacks: Any) -> bool:
    if callbacks is None:
        return False
    if isinstance(callbacks, list):
        return any(isinstance(callback, TraceEnrichmentCallback) for callback in callbacks)

    handlers = getattr(callbacks, "handlers", None)
    if isinstance(handlers, list):
        return any(isinstance(callback, TraceEnrichmentCallback) for callback in handlers)
    return False


def _with_trace_handler(callbacks: Any, trace_context: TraceContext, *, inherit: bool) -> Any:
    if _has_trace_handler(callbacks):
        return callbacks

    callback = TraceEnrichmentCallback(trace_context)
    if callbacks is None:
        return [callback]
    if isinstance(callbacks, list):
        return [*callbacks, callback]

    copied = callbacks.copy()
    copied.add_handler(callback, inherit=inherit)
    return copied


def enable_mlflow_langchain_enrichment(
    trace_context_provider: TraceContextProvider | None = None,
) -> None:
    global _PATCHED
    global _TRACE_CONTEXT_PROVIDER
    global _ORIGINAL_ASYNC_CALLBACK_CONFIGURE
    global _ORIGINAL_CALLBACK_CONFIGURE

    _TRACE_CONTEXT_PROVIDER = trace_context_provider

    with _CONFIGURE_LOCK:
        if _PATCHED:
            return

        _ORIGINAL_CALLBACK_CONFIGURE = CallbackManager.configure.__func__
        _ORIGINAL_ASYNC_CALLBACK_CONFIGURE = AsyncCallbackManager.configure.__func__

        def _wrap_configure(
            original: Callable[..., Any],
        ) -> Callable[..., Any]:
            def wrapped(
                cls: type[Any],
                inheritable_callbacks: Any = None,
                local_callbacks: Any = None,
                verbose: bool = False,
                inheritable_tags: list[str] | None = None,
                local_tags: list[str] | None = None,
                inheritable_metadata: dict[str, Any] | None = None,
                local_metadata: dict[str, Any] | None = None,
            ) -> Any:
                trace_context = _resolve_trace_context()
                if trace_context is not None and not (
                    _has_trace_handler(inheritable_callbacks) or _has_trace_handler(local_callbacks)
                ):
                    local_callbacks = _with_trace_handler(
                        local_callbacks,
                        trace_context,
                        inherit=False,
                    )

                return original(
                    cls,
                    inheritable_callbacks=inheritable_callbacks,
                    local_callbacks=local_callbacks,
                    verbose=verbose,
                    inheritable_tags=inheritable_tags,
                    local_tags=local_tags,
                    inheritable_metadata=inheritable_metadata,
                    local_metadata=local_metadata,
                )

            return wrapped

        CallbackManager.configure = classmethod(_wrap_configure(_ORIGINAL_CALLBACK_CONFIGURE))
        AsyncCallbackManager.configure = classmethod(
            _wrap_configure(_ORIGINAL_ASYNC_CALLBACK_CONFIGURE)
        )
        _PATCHED = True


def disable_mlflow_langchain_enrichment() -> None:
    global _PATCHED
    global _TRACE_CONTEXT_PROVIDER

    with _CONFIGURE_LOCK:
        if not _PATCHED:
            _TRACE_CONTEXT_PROVIDER = None
            return

        if _ORIGINAL_CALLBACK_CONFIGURE is not None:
            CallbackManager.configure = classmethod(_ORIGINAL_CALLBACK_CONFIGURE)
        if _ORIGINAL_ASYNC_CALLBACK_CONFIGURE is not None:
            AsyncCallbackManager.configure = classmethod(_ORIGINAL_ASYNC_CALLBACK_CONFIGURE)

        _TRACE_CONTEXT_PROVIDER = None
        _PATCHED = False

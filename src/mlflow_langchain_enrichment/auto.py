from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import uuid
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
_CURRENT_TRACE_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mlflow_langchain_enrichment_trace_depth",
    default=0,
)
_CURRENT_TRACE_TOKEN_USAGE_LEDGER: contextvars.ContextVar[
    dict[str, dict[str, Any]] | None
] = contextvars.ContextVar(
    "mlflow_langchain_enrichment_trace_token_usage_ledger",
    default=None,
)
_CURRENT_TRACE_TOKEN_USAGE_TOTALS: contextvars.ContextVar[
    dict[str, int] | None
] = contextvars.ContextVar(
    "mlflow_langchain_enrichment_trace_token_usage_totals",
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
    depth_token: contextvars.Token[int]
    token_usage_token: contextvars.Token[dict[str, dict[str, Any]] | None]
    token_totals_token: contextvars.Token[dict[str, int] | None]
    request: Any = None
    response: Any = None
    error: BaseException | None = None


@dataclass(slots=True)
class InvocationTraceHandle:
    trace_context: TraceContext
    span: Any
    exit_stack: contextlib.ExitStack
    depth_token: contextvars.Token[int]
    token_usage_token: contextvars.Token[dict[str, dict[str, Any]] | None] | None = None
    token_totals_token: contextvars.Token[dict[str, int] | None] | None = None
    trace_context_handle: TraceContextHandle | None = None
    is_root: bool = False
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


def get_current_trace_depth() -> int:
    return _CURRENT_TRACE_DEPTH.get()


def _push_trace_depth() -> contextvars.Token[int]:
    current_depth = get_current_trace_depth()
    return _CURRENT_TRACE_DEPTH.set(current_depth + 1)


def _reset_trace_depth(token: contextvars.Token[int]) -> None:
    _CURRENT_TRACE_DEPTH.reset(token)


def get_current_trace_token_usage_ledger() -> dict[str, dict[str, Any]] | None:
    ledger = _CURRENT_TRACE_TOKEN_USAGE_LEDGER.get()
    if ledger is None:
        return None
    return {key: dict(value) for key, value in ledger.items()}


def _push_trace_token_usage() -> contextvars.Token[dict[str, dict[str, Any]] | None]:
    current_ledger = _CURRENT_TRACE_TOKEN_USAGE_LEDGER.get()
    if current_ledger is None:
        return _CURRENT_TRACE_TOKEN_USAGE_LEDGER.set({})
    return _CURRENT_TRACE_TOKEN_USAGE_LEDGER.set(current_ledger)


def _reset_trace_token_usage(token: contextvars.Token[dict[str, dict[str, Any]] | None]) -> None:
    _CURRENT_TRACE_TOKEN_USAGE_LEDGER.reset(token)


def get_current_trace_token_usage_totals() -> dict[str, int] | None:
    totals = _CURRENT_TRACE_TOKEN_USAGE_TOTALS.get()
    if totals is None:
        return None
    return dict(totals)


def _push_trace_token_totals() -> contextvars.Token[dict[str, int] | None]:
    current_totals = _CURRENT_TRACE_TOKEN_USAGE_TOTALS.get()
    if current_totals is None:
        return _CURRENT_TRACE_TOKEN_USAGE_TOTALS.set({})
    return _CURRENT_TRACE_TOKEN_USAGE_TOTALS.set(current_totals)


def _reset_trace_token_totals(token: contextvars.Token[dict[str, int] | None]) -> None:
    _CURRENT_TRACE_TOKEN_USAGE_TOTALS.reset(token)


def accumulate_current_trace_token_usage(usage: Mapping[str, Any]) -> None:
    current = _CURRENT_TRACE_TOKEN_USAGE_TOTALS.get()
    if current is None:
        return

    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        current[key] = current.get(key, 0) + normalized


def record_current_trace_token_usage(
    usage: Mapping[str, Any],
    *,
    span_name: str | None = None,
    model_name: str | None = None,
) -> None:
    current = _CURRENT_TRACE_TOKEN_USAGE_LEDGER.get()
    if current is None:
        return

    normalized_usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        normalized_usage[key] = normalized

    if not normalized_usage:
        return

    entry_key = f"call_{uuid.uuid4().hex}"
    entry: dict[str, Any] = dict(normalized_usage)
    if span_name:
        entry["span_name"] = span_name
    if model_name:
        entry["model_name"] = model_name
    current[entry_key] = entry


def _get_aggregated_trace_token_usage() -> dict[str, int]:
    ledger = get_current_trace_token_usage_ledger() or {}
    usage = get_current_trace_token_usage_totals() or {}
    if not usage:
        for entry in ledger.values():
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = entry.get(key)
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
    return usage


def _set_span_token_usage_attributes(span: Any, usage: Mapping[str, int]) -> None:
    if not usage:
        return
    attributes: dict[str, Any] = {"mlflow.chat.tokenUsage": dict(usage)}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            attributes[key] = value

    if hasattr(span, "set_attributes"):
        span.set_attributes(attributes)
        return
    if hasattr(span, "set_attribute"):
        for key, value in attributes.items():
            span.set_attribute(key, value)


def _trace_context_with_aggregated_token_usage(trace_context: TraceContext) -> TraceContext:
    ledger = get_current_trace_token_usage_ledger() or {}
    usage = _get_aggregated_trace_token_usage()

    if not usage and not ledger:
        return trace_context

    metadata = dict(trace_context.metadata)
    metadata["mlflow.trace.tokenUsage"] = json.dumps(
        usage,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    metadata["mlflow_langchain_enrichment.tokenUsageByCall"] = json.dumps(
        ledger,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    for key, value in usage.items():
        metadata[key] = str(value)

    if metadata == dict(trace_context.metadata):
        return trace_context
    return TraceContext(
        tags=trace_context.tags,
        metadata=metadata,
        span_metadata=trace_context.span_metadata,
        run_tags=trace_context.run_tags,
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
        trace_name=trace_context.trace_name,
        capture_root_span_io=trace_context.capture_root_span_io,
    )


def _record_span_result(
    span: Any,
    trace_context: TraceContext,
    *,
    request: Any = None,
    response: Any = None,
    error: BaseException | None = None,
    update_trace: bool,
) -> None:
    if request is not None and trace_context.capture_root_span_io:
        span.set_inputs(to_root_span_inputs(request))

    if error is not None:
        if trace_context.capture_root_span_io:
            span.set_outputs({"error": f"{type(error).__name__}: {error}"})
        if update_trace:
            update_trace_from_context(
                trace_context,
                request=request,
                response=f"{type(error).__name__}: {error}",
                state="ERROR",
            )
        return

    if response is not None:
        if trace_context.capture_root_span_io:
            span.set_outputs(to_root_span_outputs(response))
        if update_trace:
            update_trace_from_context(
                trace_context,
                request=request,
                response=response,
                state="OK",
            )
        return

    if request is not None and update_trace:
        update_trace_from_context(trace_context, request=request)


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
        depth_token = _push_trace_depth()
        token_usage_token = _push_trace_token_usage()
        token_totals_token = _push_trace_token_totals()
        update_trace_from_context(resolved)
        return RootTraceHandle(
            trace_context=resolved,
            trace_context_handle=trace_context_handle,
            span=span,
            exit_stack=exit_stack,
            depth_token=depth_token,
            token_usage_token=token_usage_token,
            token_totals_token=token_totals_token,
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
    final_request = request if request is not None else handle.request
    final_error = error if error is not None else handle.error
    final_response = response if response is not None else handle.response
    final_trace_context = _trace_context_with_aggregated_token_usage(handle.trace_context)
    aggregated_usage = _get_aggregated_trace_token_usage()
    try:
        handle.trace_context = final_trace_context
        _set_span_token_usage_attributes(handle.span, aggregated_usage)
        _record_span_result(
            handle.span,
            handle.trace_context,
            request=final_request,
            response=final_response,
            error=final_error,
            update_trace=True,
        )
    finally:
        try:
            try:
                handle.exit_stack.close()
                update_trace_from_context(
                    final_trace_context,
                    request=final_request,
                    response=(
                        f"{type(final_error).__name__}: {final_error}"
                        if final_error is not None
                        else final_response
                    ),
                    state="ERROR" if final_error is not None else "OK",
                )
            finally:
                _reset_trace_depth(handle.depth_token)
                _reset_trace_token_usage(handle.token_usage_token)
                _reset_trace_token_totals(handle.token_totals_token)
        finally:
            close_trace_context(handle.trace_context_handle)


def open_traced_span(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> InvocationTraceHandle:
    resolved = _build_trace_context(trace_context, **trace_context_kwargs)

    if get_current_trace_depth() > 0 and get_current_trace_context() is not None:
        import mlflow

        exit_stack = contextlib.ExitStack()
        span = exit_stack.enter_context(mlflow.start_span(name=resolved.trace_name or "request"))
        depth_token = _push_trace_depth()
        return InvocationTraceHandle(
            trace_context=resolved,
            span=span,
            exit_stack=exit_stack,
            depth_token=depth_token,
            token_usage_token=None,
            is_root=False,
        )

    resolved_root = _build_trace_context(trace_context, **trace_context_kwargs)
    trace_context_handle = open_trace_context(resolved_root)
    exit_stack = contextlib.ExitStack()

    try:
        import mlflow

        span = exit_stack.enter_context(mlflow.start_span(name=resolved_root.trace_name or "request"))
        depth_token = _push_trace_depth()
        token_usage_token = _push_trace_token_usage()
        token_totals_token = _push_trace_token_totals()
        update_trace_from_context(resolved_root)
        return InvocationTraceHandle(
            trace_context=resolved_root,
            span=span,
            exit_stack=exit_stack,
            depth_token=depth_token,
            token_usage_token=token_usage_token,
            token_totals_token=token_totals_token,
            trace_context_handle=trace_context_handle,
            is_root=True,
        )
    except Exception:
        exit_stack.close()
        close_trace_context(trace_context_handle)
        raise


def close_traced_span(
    handle: InvocationTraceHandle,
    *,
    request: Any = None,
    response: Any = None,
    error: BaseException | None = None,
) -> None:
    final_request = request if request is not None else handle.request
    final_error = error if error is not None else handle.error
    final_response = response if response is not None else handle.response
    final_trace_context = (
        _trace_context_with_aggregated_token_usage(handle.trace_context)
        if handle.is_root
        else handle.trace_context
    )
    aggregated_usage = _get_aggregated_trace_token_usage() if handle.is_root else {}
    try:
        handle.trace_context = final_trace_context
        if handle.is_root:
            _set_span_token_usage_attributes(handle.span, aggregated_usage)
        _record_span_result(
            handle.span,
            handle.trace_context,
            request=final_request,
            response=final_response,
            error=final_error,
            update_trace=handle.is_root,
        )
    finally:
        try:
            try:
                handle.exit_stack.close()
                if handle.is_root:
                    update_trace_from_context(
                        final_trace_context,
                        request=final_request,
                        response=(
                            f"{type(final_error).__name__}: {final_error}"
                            if final_error is not None
                            else final_response
                        ),
                        state="ERROR" if final_error is not None else "OK",
                    )
            finally:
                _reset_trace_depth(handle.depth_token)
                if handle.token_usage_token is not None:
                    _reset_trace_token_usage(handle.token_usage_token)
                if handle.token_totals_token is not None:
                    _reset_trace_token_totals(handle.token_totals_token)
        finally:
            if handle.trace_context_handle is not None:
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

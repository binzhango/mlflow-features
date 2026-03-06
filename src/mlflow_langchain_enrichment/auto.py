from __future__ import annotations

import contextlib
import contextvars
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager

from .enrichment import TraceContext, TraceEnrichmentCallback


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


@contextlib.contextmanager
def using_trace_context(
    trace_context: TraceContextLike = None,
    /,
    **trace_context_kwargs: Any,
) -> Iterator[TraceContext]:
    if trace_context is not None and trace_context_kwargs:
        raise ValueError("Pass either a TraceContext object or keyword fields, not both.")

    resolved = _coerce_trace_context(trace_context)
    if resolved is None:
        resolved = TraceContext(**trace_context_kwargs)

    token = set_current_trace_context(resolved)
    run_context = contextlib.nullcontext()
    try:
        if (
            resolved.ensure_run
            or resolved.mlflow_run_name is not None
            or resolved.run_description is not None
            or resolved.run_tags
        ):
            import mlflow

            if mlflow.active_run() is None:
                run_context = mlflow.start_run(
                    run_name=resolved.mlflow_run_name or resolved.trace_name,
                    tags=dict(resolved.run_tags) or None,
                    description=resolved.run_description,
                )

        with run_context:
            yield resolved
    finally:
        reset_current_trace_context(token)


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

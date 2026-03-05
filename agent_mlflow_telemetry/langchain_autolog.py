"""Autolog-style global LangChain callback injection for custom tracing."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from .bootstrap import TelemetryRuntime, initialize_telemetry
from .config import TelemetryConfig
from .langchain_callback import CustomLangchainTracer

_InitFunc = Callable[..., None]
_MergeFunc = Callable[..., Any]


@dataclass(slots=True)
class _AutologState:
    enabled: bool = False
    installed: bool = False
    runtime: TelemetryRuntime | None = None
    run_tracer_inline: bool = True
    original_init: _InitFunc | None = None
    original_merge: _MergeFunc | None = None


_STATE = _AutologState()
_LOCK = Lock()


def autolog(
    *,
    config: TelemetryConfig | None = None,
    runtime: TelemetryRuntime | None = None,
    run_tracer_inline: bool = True,
    disable: bool = False,
) -> TelemetryRuntime | None:
    """
    Enable/disable global callback auto-injection, similar to `mlflow.langchain.autolog()`.

    When enabled, every newly created LangChain callback manager receives a
    `CustomLangchainTracer` automatically, so `.invoke()` works without passing
    `config={"callbacks": [...]}` on each call.
    """

    if disable:
        disable_autolog()
        return None

    active_runtime = runtime or initialize_telemetry(config or TelemetryConfig.from_env())
    enable_autolog(active_runtime, run_tracer_inline=run_tracer_inline)
    return active_runtime


def enable_autolog(runtime: TelemetryRuntime, *, run_tracer_inline: bool = True) -> None:
    """Enable global injection of `CustomLangchainTracer` using the provided runtime."""

    with _LOCK:
        _install_patch_if_needed()
        _STATE.runtime = runtime
        _STATE.run_tracer_inline = run_tracer_inline
        _STATE.enabled = True


def disable_autolog() -> None:
    """Disable callback auto-injection while keeping monkey patches installed."""

    with _LOCK:
        _STATE.enabled = False
        _STATE.runtime = None


def reset_autolog_state() -> None:
    """Restore original LangChain methods. Intended for tests."""

    with _LOCK:
        if _STATE.installed:
            from langchain_core.callbacks import BaseCallbackManager

            if _STATE.original_init is not None:
                BaseCallbackManager.__init__ = _STATE.original_init
            if _STATE.original_merge is not None:
                BaseCallbackManager.merge = _STATE.original_merge

        _STATE.enabled = False
        _STATE.installed = False
        _STATE.runtime = None
        _STATE.run_tracer_inline = True
        _STATE.original_init = None
        _STATE.original_merge = None


def _install_patch_if_needed() -> None:
    if _STATE.installed:
        return

    from langchain_core.callbacks import BaseCallbackManager

    original_init = BaseCallbackManager.__init__
    original_merge = BaseCallbackManager.merge

    def _patched_callback_manager_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)

        if not _STATE.enabled or _STATE.runtime is None:
            return

        handlers = getattr(self, "inheritable_handlers", [])
        for handler in handlers:
            if isinstance(handler, CustomLangchainTracer):
                return

        self.add_handler(
            CustomLangchainTracer(
                sink=_STATE.runtime.sink,
                run_inline=_STATE.run_tracer_inline,
            ),
            inherit=True,
        )

    def _patched_callback_manager_merge(self: Any, *args: Any, **kwargs: Any) -> Any:
        merged = original_merge(self, *args, **kwargs)

        if not _STATE.enabled:
            return merged

        inherited = getattr(merged, "inheritable_handlers", [])
        tracer: CustomLangchainTracer | None = None
        duplicates: list[CustomLangchainTracer] = []
        for callback in inherited:
            if not isinstance(callback, CustomLangchainTracer):
                continue
            if tracer is None:
                tracer = callback
            else:
                duplicates.append(callback)

        for duplicate in duplicates:
            merged.remove_handler(duplicate)

        return merged

    BaseCallbackManager.__init__ = _patched_callback_manager_init
    BaseCallbackManager.merge = _patched_callback_manager_merge

    _STATE.installed = True
    _STATE.original_init = original_init
    _STATE.original_merge = original_merge

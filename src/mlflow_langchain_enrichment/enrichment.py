from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - keeps importable in bare environments
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


PreviewBuilder = Callable[[Any], str | None]
DEFAULT_PREVIEW_LENGTH = 160


def _truncate(value: str, limit: int = DEFAULT_PREVIEW_LENGTH) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str, separators=(",", ":"))


def _normalize_mapping(value: Mapping[str, Any] | None) -> dict[str, str]:
    if not value:
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _message_content(message: Any) -> str:
    if isinstance(message, str):
        return message

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(message, Mapping):
        if isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(message.get("text"), str):
            return message["text"]

    return _compact_json(message)


def _flatten_messages(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return [_message_content(messages)]

    flattened: list[str] = []
    for item in messages:
        if isinstance(item, list):
            flattened.extend(_flatten_messages(item))
        else:
            flattened.append(_message_content(item))
    return [value for value in flattened if value]


def default_request_preview(request: Any, limit: int = DEFAULT_PREVIEW_LENGTH) -> str | None:
    if request is None:
        return None

    if isinstance(request, str):
        return _truncate(request.strip(), limit)

    if isinstance(request, Mapping):
        if "messages" in request:
            messages = _flatten_messages(request["messages"])
            if messages:
                return _truncate(f"{messages[0]} -> {messages[-1]}", limit)

        for key in ("question", "input", "query", "prompt"):
            value = request.get(key)
            if isinstance(value, str) and value.strip():
                return _truncate(value.strip(), limit)

        return _truncate(_compact_json(request), limit)

    if isinstance(request, list):
        messages = _flatten_messages(request)
        if messages:
            return _truncate(f"{messages[0]} -> {messages[-1]}", limit)
        return None

    return _truncate(str(request), limit)


def default_response_preview(response: Any, limit: int = DEFAULT_PREVIEW_LENGTH) -> str | None:
    if response is None:
        return None

    if isinstance(response, str):
        return _truncate(response.strip(), limit)

    if isinstance(response, Mapping):
        for key in ("output", "answer", "result", "content", "text"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return _truncate(value.strip(), limit)
        return _truncate(_compact_json(response), limit)

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return _truncate(content.strip(), limit)

    generations = getattr(response, "generations", None)
    if isinstance(generations, list) and generations:
        first_generation = generations[0]
        if isinstance(first_generation, list) and first_generation:
            text = getattr(first_generation[0], "text", None)
            if isinstance(text, str):
                return _truncate(text.strip(), limit)

    return _truncate(str(response), limit)


def _load_mlflow() -> Any:
    return importlib.import_module("mlflow")


@dataclass(slots=True)
class TraceContext:
    tags: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    span_metadata: Mapping[str, Any] = field(default_factory=dict)
    run_tags: Mapping[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    session_id: str | None = None
    client_request_id: str | None = None
    mlflow_run_name: str | None = None
    run_description: str | None = None
    ensure_run: bool = False
    request_preview: str | None = None
    response_preview: str | None = None
    request_preview_builder: PreviewBuilder | None = None
    response_preview_builder: PreviewBuilder | None = None
    preview_limit: int = DEFAULT_PREVIEW_LENGTH
    trace_name: str | None = None

    def trace_metadata(self) -> dict[str, str]:
        metadata = _normalize_mapping(self.metadata)
        if self.user_id:
            metadata.setdefault("mlflow.trace.user", self.user_id)
        if self.session_id:
            metadata.setdefault("mlflow.trace.session", self.session_id)
        return metadata

    def trace_tags(self) -> dict[str, str]:
        return _normalize_mapping(self.tags)

    def runnable_metadata(self) -> dict[str, Any]:
        metadata = dict(self.span_metadata)
        if self.trace_name:
            metadata.setdefault("trace_name", self.trace_name)
        return metadata


def update_trace_from_context(
    trace_context: TraceContext,
    *,
    request: Any = None,
    response: Any = None,
    state: str | None = None,
) -> None:
    mlflow = _load_mlflow()

    request_preview = trace_context.request_preview
    if request_preview is None and request is not None:
        if trace_context.request_preview_builder:
            request_preview = trace_context.request_preview_builder(request)
        else:
            request_preview = default_request_preview(request, trace_context.preview_limit)

    response_preview = trace_context.response_preview
    if response_preview is None and response is not None:
        if trace_context.response_preview_builder:
            response_preview = trace_context.response_preview_builder(response)
        else:
            response_preview = default_response_preview(response, trace_context.preview_limit)

    mlflow.update_current_trace(
        tags=trace_context.trace_tags() or None,
        metadata=trace_context.trace_metadata() or None,
        client_request_id=trace_context.client_request_id,
        request_preview=request_preview,
        response_preview=response_preview,
        state=state,
    )


class TraceEnrichmentCallback(BaseCallbackHandler):
    name = "mlflow-langchain-enrichment"
    raise_error = False
    run_inline = True

    def __init__(self, trace_context: TraceContext):
        self.trace_context = trace_context
        self._root_run_id: Any = None

    def _is_root(self, parent_run_id: Any) -> bool:
        return parent_run_id is None and self._root_run_id is None

    def _start_root(self, run_id: Any, request: Any) -> None:
        self._root_run_id = run_id
        update_trace_from_context(self.trace_context, request=request)

    def _finish_root(self, run_id: Any, response: Any = None, state: str | None = None) -> None:
        if run_id != self._root_run_id:
            return
        update_trace_from_context(self.trace_context, response=response, state=state)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._is_root(parent_run_id):
            self._start_root(run_id, inputs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._is_root(parent_run_id):
            self._start_root(run_id, messages)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._is_root(parent_run_id):
            self._start_root(run_id, prompts)

    def on_chain_end(self, outputs: Any, *, run_id: Any, **kwargs: Any) -> Any:
        self._finish_root(run_id, response=outputs, state="OK")

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> Any:
        self._finish_root(run_id, response=response, state="OK")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> Any:
        self._finish_root(run_id, response=f"{type(error).__name__}: {error}", state="ERROR")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> Any:
        self._finish_root(run_id, response=f"{type(error).__name__}: {error}", state="ERROR")


def build_invoke_config(
    trace_context: TraceContext,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    built = dict(config or {})

    callbacks = list(built.get("callbacks", []))
    if not any(isinstance(callback, TraceEnrichmentCallback) for callback in callbacks):
        callbacks.append(TraceEnrichmentCallback(trace_context))
    built["callbacks"] = callbacks

    metadata = dict(built.get("metadata", {}))
    metadata.update(trace_context.runnable_metadata())
    if metadata:
        built["metadata"] = metadata

    if trace_context.trace_name and "run_name" not in built:
        built["run_name"] = trace_context.trace_name

    return built


def invoke_with_enrichment(
    runnable: Any,
    inputs: Any,
    trace_context: TraceContext,
    *,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    return runnable.invoke(inputs, config=build_invoke_config(trace_context, config), **kwargs)


async def ainvoke_with_enrichment(
    runnable: Any,
    inputs: Any,
    trace_context: TraceContext,
    *,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    return await runnable.ainvoke(inputs, config=build_invoke_config(trace_context, config), **kwargs)

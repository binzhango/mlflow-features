from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_mlflow_telemetry.config import TelemetryConfig
from agent_mlflow_telemetry.langchain_callback import TelemetryCallbackHandler
from agent_mlflow_telemetry.llmclient_adapter import InstrumentedLLMClient, wrap_llmclient
from agent_mlflow_telemetry.schema import SpanRecord


class RecordingSink:
    def __init__(self) -> None:
        self.config = TelemetryConfig()
        self.started: list[SpanRecord] = []
        self.ended: list[SpanRecord] = []

    def start_span(self, span: SpanRecord) -> None:
        self.started.append(span)

    def end_span(self, span: SpanRecord) -> None:
        self.ended.append(span)

    def record_event(self, span_id: str, name: str, attributes: dict[str, object] | None = None) -> None:
        _ = (span_id, name, attributes)


@dataclass
class _Resp:
    content: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    usage_metadata: dict[str, int] | None = None
    response_metadata: dict[str, object] | None = None
    additional_kwargs: dict[str, object] | None = None


class DummyOllamaClient:
    model = "glm-4.7-flash"
    gateway_route = "/gateway"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def invoke(self, *args: object, **kwargs: object) -> _Resp:
        self.calls.append(("invoke", args, kwargs))
        return _Resp(
            usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            response_metadata={"model_name": "glm-4.7-flash"},
            additional_kwargs={"finish_reason": "stop"},
        )

    async def ainvoke(self, *args: object, **kwargs: object) -> _Resp:
        self.calls.append(("ainvoke", args, kwargs))
        return _Resp(
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            response_metadata={"model_name": "glm-4.7-flash"},
        )

    def stream(self, *args: object, **kwargs: object):
        self.calls.append(("stream", args, kwargs))
        for chunk in ["c1", "c2"]:
            yield chunk


class DummyFailClient:
    model = "glm-4.7-flash"

    def invoke(self, *args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise RuntimeError("invoke failed")


class DummyToolCallClient:
    model = "nemotron-3-nano"

    def invoke(self, *args: object, **kwargs: object) -> _Resp:
        _ = (args, kwargs)
        return _Resp(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "skill",
                    "function": {"name": "summarize", "arguments": "{\"topic\":\"mlflow\"}"},
                }
            ],
            response_metadata={"model_name": "nemotron-3-nano", "done_reason": "tool_call"},
        )


def test_callback_methods_without_run_id_are_safe_noops() -> None:
    handler = TelemetryCallbackHandler(sink=RecordingSink())

    assert handler.on_chain_start({"id": "chain"}, {"input": "x"}) is None
    assert handler.on_chain_end({"output": "y"}) is None
    assert handler.on_chain_error(RuntimeError("boom")) is None


def test_wrap_llmclient_returns_instrumented_wrapper() -> None:
    wrapped = wrap_llmclient(DummyOllamaClient(), RecordingSink())
    assert isinstance(wrapped, InstrumentedLLMClient)


def test_instrumented_client_sync_and_stream_emit_spans() -> None:
    client = DummyOllamaClient()
    sink = RecordingSink()
    wrapped = InstrumentedLLMClient(client, sink)

    sync_result = wrapped.invoke("hello", temperature=0, reasoning={"effort": "medium"})
    stream_result = list(wrapped.stream("hello", stream=True))

    assert isinstance(sync_result, _Resp)
    assert stream_result == ["c1", "c2"]
    assert client.calls[0] == ("invoke", ("hello",), {"temperature": 0, "reasoning": {"effort": "medium"}})
    assert client.calls[1] == ("stream", ("hello",), {"stream": True})

    assert len(sink.started) == 2
    assert len(sink.ended) == 2
    assert sink.ended[0].status == "ok"
    assert sink.ended[0].model_name == "glm-4.7-flash"
    assert sink.ended[0].provider == "ollama"
    assert sink.ended[0].total_tokens == 5
    assert sink.ended[0].attributes["response_metadata"] == {"model_name": "glm-4.7-flash"}
    assert sink.ended[0].attributes["response_metadata.model_name"] == "glm-4.7-flash"
    assert sink.ended[0].attributes["model_config.temperature"] == 0
    assert sink.ended[0].attributes["model_config.reasoning_effort"] == "medium"
    assert sink.started[0].attributes["_mlflow_inputs"]["temperature"] == 0
    assert sink.started[0].attributes["_mlflow_inputs"]["reasoning_effort"] == "medium"
    assert sink.ended[0].attributes["additional_kwargs"] == {"finish_reason": "stop"}
    assert sink.ended[0].attributes["usage_metadata"] == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    assert sink.ended[0].attributes["_mlflow_outputs"] == {
        "messages": [{"role": "assistant", "content": ""}],
        "model": "glm-4.7-flash",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "response_metadata": {"model_name": "glm-4.7-flash"},
    }
    assert sink.ended[0].attributes["mlflow.chat.messages"][0] == {
        "role": "assistant",
        "content": "",
    }
    assert sink.ended[1].streamed is True


@pytest.mark.asyncio
async def test_instrumented_client_forwards_async_calls() -> None:
    client = DummyOllamaClient()
    sink = RecordingSink()
    wrapped = InstrumentedLLMClient(client, sink)

    async_result = await wrapped.ainvoke("hello", timeout=3)

    assert isinstance(async_result, _Resp)
    assert client.calls[0] == ("ainvoke", ("hello",), {"timeout": 3})
    assert len(sink.started) == 1
    assert len(sink.ended) == 1
    assert sink.ended[0].status == "ok"
    assert sink.ended[0].total_tokens == 2


def test_instrumented_client_error_path_emits_error_span() -> None:
    sink = RecordingSink()
    wrapped = InstrumentedLLMClient(DummyFailClient(), sink)

    with pytest.raises(RuntimeError, match="invoke failed"):
        wrapped.invoke("hello")

    assert len(sink.started) == 1
    assert len(sink.ended) == 1
    assert sink.ended[0].status == "error"
    assert sink.ended[0].error_type == "RuntimeError"


def test_adapter_extracts_token_usage_from_response_metadata_eval_counts() -> None:
    usage = InstrumentedLLMClient._extract_token_usage(
        _Resp(response_metadata={"prompt_eval_count": 22, "eval_count": 2596})
    )
    assert usage == {"input_tokens": 22, "output_tokens": 2596, "total_tokens": 2618}


def test_instrumented_client_emits_child_tool_call_span() -> None:
    sink = RecordingSink()
    wrapped = InstrumentedLLMClient(DummyToolCallClient(), sink)

    _ = wrapped.invoke("hello")

    tool_started = [s for s in sink.started if s.component == "tool"]
    tool_ended = [s for s in sink.ended if s.component == "tool"]
    assert len(tool_started) == 1
    assert len(tool_ended) == 1
    assert tool_started[0].operation == "tool_call.skill.summarize"
    assert tool_ended[0].attributes["tool_call_id"] == "call_1"


def test_adapter_flattens_nested_response_metadata() -> None:
    sink = RecordingSink()
    wrapped = InstrumentedLLMClient(DummyOllamaClient(), sink)

    _ = wrapped.invoke("hello")

    attrs = sink.ended[0].attributes
    assert attrs["response_metadata.model_name"] == "glm-4.7-flash"

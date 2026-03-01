from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from agent_mlflow_telemetry.config import TelemetryConfig
from agent_mlflow_telemetry.context import with_trace_context
from agent_mlflow_telemetry.langchain_callback import TelemetryCallbackHandler
from agent_mlflow_telemetry.schema import REQUIRED_ATTRIBUTE_KEYS, SpanRecord


class RecordingSink:
    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig()
        self.started: list[SpanRecord] = []
        self.ended: list[SpanRecord] = []

    def start_span(self, span: SpanRecord) -> None:
        self.started.append(span)

    def end_span(self, span: SpanRecord) -> None:
        self.ended.append(span)

    def record_event(self, span_id: str, name: str, attributes: dict[str, object] | None = None) -> None:
        _ = (span_id, name, attributes)


@dataclass
class _Msg:
    content: str
    tool_calls: list[dict[str, object]] | None = None
    response_metadata: dict[str, object] | None = None
    additional_kwargs: dict[str, object] | None = None
    usage_metadata: dict[str, int] | None = None


@dataclass
class _Gen:
    text: str = ""
    message: _Msg | None = None


@dataclass
class _Resp:
    generations: list[list[_Gen]]
    llm_output: dict[str, object]


def test_callback_creates_trace_tree_and_populates_llm_metrics() -> None:
    sink = RecordingSink(TelemetryConfig(service_name="svc", environment="test"))
    cb = TelemetryCallbackHandler(sink=sink)

    chain_run = uuid4()
    llm_run = uuid4()

    cb.on_chain_start({"name": "agent_chain"}, {"input": "hello"}, run_id=chain_run, parent_run_id=None)
    cb.on_llm_start(
        {"name": "ChatOllama"},
        ["hello"],
        run_id=llm_run,
        parent_run_id=chain_run,
        invocation_params={
            "model": "glm-4.7-flash",
            "gateway_route": "/gateway",
            "temperature": 0.2,
            "reasoning": {"effort": "high"},
        },
    )
    cb.on_llm_end(
        _Resp(
            generations=[[ _Gen(text="world") ]],
            llm_output={
                "token_usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                "model_name": "glm-4.7-flash",
            },
        ),
        run_id=llm_run,
        parent_run_id=chain_run,
    )
    cb.on_chain_end({"output": "done"}, run_id=chain_run, parent_run_id=None)

    assert len(sink.started) == 2
    assert len(sink.ended) == 2

    started_chain, started_llm = sink.started
    assert started_chain.span_id == str(chain_run)
    assert started_llm.parent_span_id == str(chain_run)
    assert started_llm.attributes["_mlflow_inputs"]["temperature"] == 0.2
    assert started_llm.attributes["_mlflow_inputs"]["reasoning_effort"] == "high"

    ended_llm = [s for s in sink.ended if s.span_id == str(llm_run)][0]
    assert ended_llm.input_tokens == 4
    assert ended_llm.output_tokens == 6
    assert ended_llm.total_tokens == 10
    assert ended_llm.model_name == "glm-4.7-flash"
    assert ended_llm.provider == "ollama"
    assert ended_llm.latency_ms is not None and ended_llm.latency_ms >= 0
    assert "response_metadata" in ended_llm.attributes
    assert ended_llm.attributes["response_metadata.model_name"] == "glm-4.7-flash"
    assert ended_llm.attributes["model_config.temperature"] == 0.2
    assert ended_llm.attributes["model_config.reasoning_effort"] == "high"
    outputs = ended_llm.attributes["_mlflow_outputs"]
    assert outputs["messages"] == [{"role": "assistant", "content": "world"}]
    assert outputs["model"] == "glm-4.7-flash"
    assert outputs["usage"] == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    assert outputs["response_metadata"]["model_name"] == "glm-4.7-flash"
    assert ended_llm.attributes["mlflow.chat.messages"][0]["role"] == "assistant"


def test_callback_extracts_token_usage_from_message_usage_metadata() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)

    llm_run = uuid4()
    cb.on_llm_start(
        {"name": "ChatOllama"},
        ["hello"],
        run_id=llm_run,
        parent_run_id=None,
        invocation_params={"model": "glm-4.7-flash"},
    )
    cb.on_llm_end(
        _Resp(
            generations=[
                [
                    _Gen(
                        message=_Msg(
                            content="world",
                            usage_metadata={"input_tokens": 7, "output_tokens": 11, "total_tokens": 18},
                        )
                    )
                ]
            ],
            llm_output={},
        ),
        run_id=llm_run,
        parent_run_id=None,
    )

    ended = sink.ended[0]
    assert ended.input_tokens == 7
    assert ended.output_tokens == 11
    assert ended.total_tokens == 18


def test_callback_extracts_token_usage_from_response_metadata_eval_counts() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)

    llm_run = uuid4()
    cb.on_llm_start(
        {"name": "ChatOllama"},
        ["hello"],
        run_id=llm_run,
        parent_run_id=None,
        invocation_params={"model": "nemotron-3-nano"},
    )
    cb.on_llm_end(
        _Resp(
            generations=[
                [
                    _Gen(
                        message=_Msg(
                            content="telemetry-ready",
                            response_metadata={"prompt_eval_count": 22, "eval_count": 7},
                        )
                    )
                ]
            ],
            llm_output={},
        ),
        run_id=llm_run,
        parent_run_id=None,
    )

    ended = sink.ended[0]
    assert ended.input_tokens == 22
    assert ended.output_tokens == 7
    assert ended.total_tokens == 29
    assert ended.attributes["response_metadata.prompt_eval_count"] == 22


def test_callback_emits_child_tool_call_spans_from_llm_output() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)

    llm_run = uuid4()
    cb.on_llm_start(
        {"name": "ChatOllama"},
        ["hello"],
        run_id=llm_run,
        parent_run_id=None,
        invocation_params={"model": "nemotron-3-nano"},
    )
    cb.on_llm_end(
        _Resp(
            generations=[
                [
                    _Gen(
                        message=_Msg(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call_1",
                                    "type": "mcp",
                                    "function": {"name": "search_docs", "arguments": "{\"q\":\"mlflow\"}"},
                                }
                            ],
                            response_metadata={"done_reason": "tool_call"},
                        )
                    )
                ]
            ],
            llm_output={},
        ),
        run_id=llm_run,
        parent_run_id=None,
    )

    tool_started = [s for s in sink.started if s.component == "tool"]
    tool_ended = [s for s in sink.ended if s.component == "tool"]
    assert len(tool_started) == 1
    assert len(tool_ended) == 1
    assert tool_started[0].parent_span_id == str(llm_run)
    assert tool_started[0].operation == "tool_call.mcp.search_docs"
    assert tool_ended[0].attributes["tool_call_id"] == "call_1"


def test_callback_error_path_emits_error_span() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)

    tool_run = uuid4()
    cb.on_tool_start({"name": "search"}, "pizza", run_id=tool_run, parent_run_id=None)
    cb.on_tool_error(RuntimeError("tool failed"), run_id=tool_run, parent_run_id=None)

    assert len(sink.ended) == 1
    ended = sink.ended[0]
    assert ended.status == "error"
    assert ended.error_type == "RuntimeError"
    assert "tool failed" in (ended.error_message or "")


def test_regression_no_empty_required_attribute_columns() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)

    run_id = uuid4()
    cb.on_chain_start({}, {"input": "x"}, run_id=run_id, parent_run_id=None)
    cb.on_chain_end({"output": "y"}, run_id=run_id, parent_run_id=None)

    assert len(sink.ended) == 1
    attrs = sink.ended[0].to_attributes()
    for key in REQUIRED_ATTRIBUTE_KEYS:
        assert key in attrs
        assert attrs[key] not in (None, "")


def test_callback_uses_upstream_trace_context_for_root_span() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)
    run_id = uuid4()

    with with_trace_context(trace_id="upstream-trace", span_id="upstream-parent"):
        cb.on_chain_start({"name": "root"}, {"input": "x"}, run_id=run_id, parent_run_id=None)
        cb.on_chain_end({"output": "y"}, run_id=run_id, parent_run_id=None)

    assert sink.started[0].trace_id == "upstream-trace"
    assert sink.started[0].parent_span_id == "upstream-parent"


def test_chain_outputs_are_chat_formatted_for_plain_response_preview() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)
    run_id = uuid4()

    cb.on_chain_start({"name": "root"}, {"input": "x"}, run_id=run_id, parent_run_id=None)
    cb.on_chain_end("In Boston it is 6C and cloudy.", run_id=run_id, parent_run_id=None)

    ended = sink.ended[0]
    assert ended.attributes["response_text"] == "In Boston it is 6C and cloudy."
    assert ended.attributes["_mlflow_outputs"]["messages"] == [
        {"role": "assistant", "content": "In Boston it is 6C and cloudy."}
    ]
    assert ended.attributes["mlflow.chat.messages"][0]["content"] == "In Boston it is 6C and cloudy."


def test_callback_extracts_model_config_from_serialized_repr() -> None:
    sink = RecordingSink()
    cb = TelemetryCallbackHandler(sink=sink)
    llm_run = uuid4()

    cb.on_llm_start(
        {
            "name": "ChatOllama",
            "repr": "ChatOllama(model='nemotron-3-nano', temperature=0.4, reasoning={'effort': 'high'})",
        },
        ["hello"],
        run_id=llm_run,
        parent_run_id=None,
        invocation_params={"model": "nemotron-3-nano"},
    )
    cb.on_llm_end(
        _Resp(
            generations=[[_Gen(text="world")]],
            llm_output={},
        ),
        run_id=llm_run,
        parent_run_id=None,
    )

    ended = sink.ended[0]
    assert ended.attributes["model_config.temperature"] == 0.4
    assert ended.attributes["model_config.reasoning_effort"] == "high"
    assert sink.started[0].attributes["_mlflow_inputs"]["temperature"] == 0.4
    assert sink.started[0].attributes["_mlflow_inputs"]["reasoning_effort"] == "high"

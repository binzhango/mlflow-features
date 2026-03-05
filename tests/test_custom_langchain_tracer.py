from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from mlflow.entities import SpanType
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from mlflow.tracing.constant import SpanAttributeKey

from agent_mlflow_telemetry.langchain_callback import CustomLangchainTracer


class _FakeSpan:
    def __init__(self, *, span_type: str, attributes: dict[str, object] | None = None) -> None:
        self.span_type = span_type
        self.attributes = dict(attributes or {})
        self.ended: list[dict[str, object]] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def end(self, outputs: object | None = None, attributes: dict[str, object] | None = None, status: object | None = None) -> None:
        self.ended.append({"outputs": outputs, "attributes": attributes or {}, "status": status})


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


def test_custom_tracer_enriches_llm_start_attributes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_super_start(self, span_name, parent_run_id, span_type, run_id, inputs=None, attributes=None):  # noqa: ANN001
        captured["span_name"] = span_name
        captured["span_type"] = span_type
        captured["run_id"] = str(run_id)
        captured["inputs"] = inputs
        captured["attributes"] = attributes or {}
        return _FakeSpan(span_type=span_type)

    monkeypatch.setattr(MlflowLangchainTracer, "_start_span", _fake_super_start)

    run_id = uuid4()
    tracer = CustomLangchainTracer(static_attributes={"team": "mlflow"}, log_content=True, run_inline=True)
    invocation_params = {
        "model": "nemotron-3-nano",
        "temperature": 0.2,
        "reasoning": {"effort": "high"},
    }
    tracer._remember_llm_run_state(
        run_id=run_id,
        serialized={
            "name": "ChatOllama",
            "repr": "ChatOllama(model='nemotron-3-nano', temperature=0.4, reasoning={'effort': 'high'})",
        },
        invocation_params=invocation_params,
        metadata={},
    )

    span = tracer._start_span(
        span_name="ChatOllama",
        parent_run_id=None,
        span_type=SpanType.CHAT_MODEL,
        run_id=run_id,
        inputs=["hello"],
        attributes={"invocation_params": invocation_params},
    )

    attrs = captured["attributes"]
    assert captured["inputs"]["messages"][0]["content"] == "hello"
    assert captured["inputs"]["temperature"] == 0.2
    assert captured["inputs"]["reasoning_effort"] == "high"
    assert attrs[SpanAttributeKey.MESSAGE_FORMAT] == "openai"
    assert attrs["prompt_text"] == "hello"
    assert attrs["prompt_hash"]
    assert attrs["model_config.temperature"] == 0.2
    assert attrs["model_config.reasoning_effort"] == "high"
    assert span.attributes["team"] == "mlflow"
    assert span.attributes[SpanAttributeKey.MODEL] == "nemotron-3-nano"
    assert span.attributes[SpanAttributeKey.MODEL_PROVIDER] == "ollama"


def test_custom_tracer_enriches_llm_end_and_emits_tool_call_spans(monkeypatch) -> None:
    end_captured: dict[str, object] = {}
    child_spans: list[dict[str, object]] = []

    def _fake_super_end(self, run_id, span, outputs=None, attributes=None, status=None):  # noqa: ANN001
        end_captured["run_id"] = str(run_id)
        end_captured["outputs"] = outputs
        end_captured["attributes"] = attributes or {}
        end_captured["status"] = status

    def _fake_start_span_no_context(  # noqa: ANN001
        name,
        span_type="UNKNOWN",
        parent_span=None,
        inputs=None,
        attributes=None,
        tags=None,
        metadata=None,
        experiment_id=None,
        start_time_ns=None,
    ):
        _ = (tags, metadata, experiment_id, start_time_ns)
        child = _FakeSpan(span_type=span_type, attributes=attributes)
        child_spans.append(
            {
                "name": name,
                "span_type": span_type,
                "parent_span": parent_span,
                "inputs": inputs,
                "span": child,
            }
        )
        return child

    monkeypatch.setattr(MlflowLangchainTracer, "_end_span", _fake_super_end)
    monkeypatch.setattr("agent_mlflow_telemetry.langchain_callback.start_span_no_context", _fake_start_span_no_context)

    run_id = uuid4()
    tracer = CustomLangchainTracer(log_content=True, run_inline=True)
    tracer._run_state[str(run_id)] = {
        "model_config": {"temperature": 0.3, "reasoning_effort": "medium"},
        "model_name": "nemotron-3-nano",
        "provider": "ollama",
    }
    parent_span = _FakeSpan(
        span_type=SpanType.CHAT_MODEL,
        attributes={SpanAttributeKey.MODEL: "nemotron-3-nano"},
    )
    response = _Resp(
        generations=[
            [
                _Gen(
                    message=_Msg(
                        content="use tool",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup_weather", "arguments": "{\"city\":\"Boston\"}"},
                            }
                        ],
                        response_metadata={"model_name": "nemotron-3-nano"},
                        additional_kwargs={"finish_reason": "tool_calls"},
                    )
                )
            ]
        ],
        llm_output={"token_usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}},
    )

    tracer._end_span(run_id=run_id, span=parent_span, outputs=response)

    attrs = end_captured["attributes"]
    assert attrs[SpanAttributeKey.MESSAGE_FORMAT] == "openai"
    assert attrs["response_hash"]
    assert attrs["response_text"] == "use tool"
    assert attrs["model_config.temperature"] == 0.3
    assert attrs["response_metadata.model_name"] == "nemotron-3-nano"
    assert attrs["additional_kwargs"] == {"finish_reason": "tool_calls"}
    assert attrs[SpanAttributeKey.CHAT_USAGE] == {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}
    assert attrs["mlflow.chat.messages"][0]["role"] == "assistant"

    outputs = end_captured["outputs"]
    assert outputs["messages"][0]["content"] == "use tool"
    assert outputs["usage"] == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}

    assert len(child_spans) == 1
    child = child_spans[0]
    assert child["name"] == "tool_call.function.lookup_weather"
    assert child["span_type"] == SpanType.TOOL
    assert child["inputs"]["name"] == "lookup_weather"
    assert child["span"].ended[0]["attributes"]["tool_call_id"] == "call_1"
    assert child["span"].ended[0]["outputs"]["tool_call"]["function"]["name"] == "lookup_weather"
    assert str(run_id) not in tracer._run_state

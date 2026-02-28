from __future__ import annotations

from dataclasses import dataclass

from agent_mlflow_telemetry.chat_payloads import (
    build_chat_messages,
    build_chat_outputs,
    build_chat_request,
    build_chat_response,
    request_preview_text,
    response_preview_text,
)


@dataclass
class _Msg:
    type: str
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
    generations: list[list[_Gen]] | None = None


def test_build_chat_request_from_plain_text() -> None:
    payload = build_chat_request("Say hello")
    assert payload == {"messages": [{"role": "user", "content": "Say hello"}]}
    assert request_preview_text(payload) == "Say hello"


def test_build_chat_request_from_langchain_like_messages() -> None:
    payload = build_chat_request([[_Msg(type="human", content="Hi"), _Msg(type="system", content="Rules")]])
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hi"
    assert payload["messages"][1]["role"] == "system"


def test_build_chat_response_from_ai_like_message() -> None:
    payload = build_chat_response(_Msg(type="ai", content="Hello there world"))
    assert payload["role"] == "assistant"
    assert payload["content"] == "Hello there world"
    assert response_preview_text(payload) == "Hello there world"


def test_build_chat_response_preserves_generation_message_metadata() -> None:
    payload = build_chat_response(
        _Resp(
            generations=[
                [
                    _Gen(
                        message=_Msg(
                            type="ai",
                            content="telemetry-ready",
                            response_metadata={
                                "prompt_eval_count": 22,
                                "eval_count": 3,
                                "model_name": "nemotron-3-nano",
                            },
                            additional_kwargs={"finish_reason": "stop"},
                            usage_metadata={"input_tokens": 22, "output_tokens": 3, "total_tokens": 25},
                        )
                    )
                ]
            ]
        )
    )
    assert payload["role"] == "assistant"
    assert payload["content"] == "telemetry-ready"
    assert set(payload.keys()) == {"role", "content"}


def test_build_chat_response_keeps_tool_only_assistant_message() -> None:
    payload = build_chat_response(
        _Resp(
            generations=[
                [
                    _Gen(
                        message=_Msg(
                            type="ai",
                            content="",
                            tool_calls=[
                                {
                                    "id": "call_1",
                                    "type": "mcp",
                                    "function": {"name": "search_docs", "arguments": "{\"q\":\"mlflow\"}"},
                                }
                            ],
                        )
                    )
                ]
            ]
        )
    )
    assert payload["role"] == "assistant"
    assert payload["content"] == ""


def test_build_chat_messages_returns_simple_assistant_message() -> None:
    messages = build_chat_messages(_Msg(type="ai", content="Hello there world"), default_role="assistant")
    assert messages == [{"role": "assistant", "content": "Hello there world"}]


def test_build_chat_outputs_uses_messages_shape_and_preserves_metadata() -> None:
    payload = build_chat_outputs(
        _Msg(
            type="ai",
            content="Hello there world",
            response_metadata={"model_name": "nemotron-3-nano", "done_reason": "stop"},
            usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        ),
        model_name="nemotron-3-nano",
        token_usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    assert payload["messages"] == [{"role": "assistant", "content": "Hello there world"}]
    assert payload["model"] == "nemotron-3-nano"
    assert payload["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert payload["response_metadata"]["model_name"] == "nemotron-3-nano"

"""Helpers for building chat-friendly span inputs/outputs."""

from __future__ import annotations

import json
from typing import Any


_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def to_text(payload: Any) -> str:
    """Return a stable text representation for hashing and preview fields."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        if "content" in payload:
            return _content_to_text(payload.get("content"))
        if "messages" in payload and isinstance(payload["messages"], list):
            return _messages_preview(payload["messages"], role="user")
        if "output" in payload and isinstance(payload["output"], dict):
            return _content_to_text(payload["output"].get("content"))
        if "output" in payload and isinstance(payload["output"], list):
            return _messages_preview(payload["output"], role="assistant")
        if "choices" in payload and isinstance(payload["choices"], list):
            return _choices_preview(payload["choices"], role="assistant")

    if hasattr(payload, "content"):
        return _content_to_text(getattr(payload, "content"))

    return _json_or_str(payload)


def build_chat_request(payload: Any) -> dict[str, Any]:
    """Build OpenAI-style chat request payload for span inputs."""
    messages = _extract_messages(payload, default_role="user")
    if messages:
        return {"messages": messages}
    return {"messages": [{"role": "user", "content": to_text(payload)}]}


def build_chat_response(payload: Any) -> dict[str, Any]:
    """Build assistant-centric chat response payload for span outputs."""
    messages = _extract_messages(payload, default_role="assistant")
    if messages:
        primary = _primary_assistant_message(messages)
        response_payload: dict[str, Any] = {
            "role": str(primary.get("role", "assistant")),
            "content": _content_to_text(primary.get("content")),
        }
    else:
        response_payload = {"role": "assistant", "content": to_text(payload)}
    return response_payload


def build_chat_messages(payload: Any, *, default_role: str = "assistant") -> list[dict[str, Any]]:
    """Build chat message list for mlflow.chat.messages UI rendering."""
    messages = _extract_messages(payload, default_role=default_role)
    if not messages:
        fallback = _content_to_text(payload)
        return [{"role": default_role, "content": fallback}]

    compact_messages: list[dict[str, Any]] = []
    for msg in messages:
        compact = {
            "role": str(msg.get("role", default_role)),
            "content": _content_to_text(msg.get("content")),
        }
        if isinstance(msg.get("tool_calls"), list):
            compact["tool_calls"] = _json_safe(msg["tool_calls"])
        compact_messages.append(compact)
    return compact_messages


def build_chat_outputs(
    payload: Any,
    *,
    default_role: str = "assistant",
    model_name: str | None = None,
    token_usage: dict[str, int] | None = None,
    response_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a UI-friendly chat outputs payload for MLflow spans."""
    messages = build_chat_messages(payload, default_role=default_role)
    output_payload: dict[str, Any] = {"messages": messages}

    model = model_name or _response_model(payload, messages)
    if model:
        output_payload["model"] = model

    usage = _usage_from_tokens(token_usage) if token_usage else None
    if usage is None:
        usage = _response_usage(payload, messages)
    if usage:
        output_payload["usage"] = usage

    metadata = response_metadata or _response_metadata(payload, messages)
    if metadata:
        output_payload["response_metadata"] = _json_safe(metadata)
    return output_payload


def extract_tool_calls(payload: Any) -> list[dict[str, Any]]:
    """Extract normalized tool-call rows from raw or normalized response payload."""
    tool_calls: list[dict[str, Any]] = []
    messages = _extract_messages(payload, default_role="assistant")
    if not messages and isinstance(payload, dict):
        messages = [m for m in _output_messages(payload) if isinstance(m, dict)]
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call_index, raw_call in enumerate(calls):
            call = raw_call if isinstance(raw_call, dict) else {"value": _json_safe(raw_call)}
            function = call.get("function")
            function_dict = function if isinstance(function, dict) else {}
            name = call.get("name") or function_dict.get("name") or call.get("tool_name")
            arguments = (
                call.get("arguments")
                if "arguments" in call
                else function_dict.get("arguments", call.get("args"))
            )
            call_id = call.get("id") or call.get("tool_call_id")
            call_type = call.get("type") or "tool"
            normalized = {
                "id": str(call_id) if call_id is not None else None,
                "type": str(call_type),
                "name": str(name) if name is not None else "unknown",
                "arguments": _json_safe(arguments),
                "raw": _json_safe(call),
                "message_index": message_index,
                "call_index": call_index,
            }
            tool_calls.append(normalized)
    return tool_calls


def request_preview_text(request_payload: dict[str, Any]) -> str:
    if isinstance(request_payload.get("messages"), list):
        return _messages_preview(request_payload["messages"], role="user")
    return to_text(request_payload)


def response_preview_text(response_payload: Any) -> str:
    if isinstance(response_payload, list):
        return _messages_preview(response_payload, role="assistant")
    if isinstance(response_payload, dict):
        if isinstance(response_payload.get("choices"), list):
            return _choices_preview(response_payload["choices"], role="assistant")
        if response_payload.get("role") == "assistant" and "content" in response_payload:
            return _content_to_text(response_payload.get("content"))
        if isinstance(response_payload.get("output"), list):
            return _messages_preview(response_payload["output"], role="assistant")
        if isinstance(response_payload.get("output"), dict):
            return _content_to_text(response_payload["output"].get("content"))
    return to_text(response_payload)


def _extract_messages(payload: Any, *, default_role: str) -> list[dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, str):
        return [{"role": default_role, "content": payload}]

    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            return [msg for item in payload["messages"] if (msg := _message_from_item(item, default_role=default_role))]
        if isinstance(payload.get("input"), list):
            return [msg for item in payload["input"] if (msg := _message_from_item(item, default_role=default_role))]
        if isinstance(payload.get("output"), list):
            return [msg for item in payload["output"] if (msg := _message_from_item(item, default_role=default_role))]
        if isinstance(payload.get("choices"), list):
            extracted: list[dict[str, Any]] = []
            for choice in payload["choices"]:
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message")
                parsed = _message_from_item(msg, default_role=default_role)
                if parsed:
                    extracted.append(parsed)
            return extracted
        if msg := _message_from_item(payload, default_role=default_role):
            return [msg]
        return []

    if isinstance(payload, list):
        items = payload
        if items and isinstance(items[0], list):
            items = items[0]
        messages = [msg for item in items if (msg := _message_from_item(item, default_role=default_role))]
        return messages

    if hasattr(payload, "generations"):
        messages: list[dict[str, Any]] = []
        for group in getattr(payload, "generations", []) or []:
            for item in group:
                if msg := _message_from_item(item, default_role=default_role):
                    messages.append(msg)
        return messages

    if msg := _message_from_item(payload, default_role=default_role):
        return [msg]

    return []


def _message_from_item(item: Any, *, default_role: str) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, str):
        return {"role": default_role, "content": item}

    if isinstance(item, dict):
        role = _normalize_role(item.get("role"), default=default_role)
        content = _content_to_text(item.get("content", item.get("text")))
        payload: dict[str, Any] = {"role": role, "content": content}
        for key in ("tool_calls", "additional_kwargs", "response_metadata", "usage_metadata"):
            if key in item and item[key] is not None:
                payload[key] = _json_safe(item[key])
        if content or any(k in payload for k in ("tool_calls", "additional_kwargs", "response_metadata", "usage_metadata")):
            return payload
        return None

    # LangChain generation objects often wrap the actual message at `.message`.
    nested_message = getattr(item, "message", None)
    if nested_message is not None and nested_message is not item:
        nested = _message_from_item(nested_message, default_role=default_role)
        if nested is not None:
            if not _content_to_text(nested.get("content")):
                fallback_content = _content_to_text(getattr(item, "text", None))
                if fallback_content:
                    nested["content"] = fallback_content
            for key in ("tool_calls", "additional_kwargs", "response_metadata", "usage_metadata"):
                if key in nested:
                    continue
                value = getattr(item, key, None)
                if value is not None:
                    nested[key] = _json_safe(value)
            return nested

    role = _normalize_role(getattr(item, "role", None) or getattr(item, "type", None), default=default_role)
    content = _content_to_text(getattr(item, "content", None) or getattr(item, "text", None))
    payload = {"role": role, "content": content}
    for key in ("tool_calls", "additional_kwargs", "response_metadata", "usage_metadata"):
        value = getattr(item, key, None)
        if value is not None:
            payload[key] = _json_safe(value)
    if content or any(k in payload for k in ("tool_calls", "additional_kwargs", "response_metadata", "usage_metadata")):
        return payload
    return None


def _normalize_role(raw: Any, *, default: str) -> str:
    if raw is None:
        return default
    lowered = str(raw).strip().lower()
    return _ROLE_ALIASES.get(lowered, lowered or default)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                if part.get("type") in {"text", "output_text"} and part.get("text"):
                    parts.append(str(part["text"]))
                    continue
                if part.get("content") is not None:
                    parts.append(_content_to_text(part.get("content")))
                    continue
        if parts:
            return "\n".join(parts)

    return _json_or_str(content)


def _messages_preview(messages: list[Any], *, role: str) -> str:
    for item in reversed(messages):
        msg = _message_from_item(item, default_role=role)
        if msg and msg.get("role") == role:
            return _content_to_text(msg.get("content"))
    if messages:
        msg = _message_from_item(messages[-1], default_role=role)
        if msg:
            return _content_to_text(msg.get("content"))
    return ""


def _choices_preview(choices: list[Any], *, role: str) -> str:
    messages: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if msg := _message_from_item(choice.get("message"), default_role=role):
            messages.append(msg)
    return _messages_preview(messages, role=role)


def _output_messages(response_payload: dict[str, Any]) -> list[Any]:
    if isinstance(response_payload.get("role"), str):
        return [response_payload]
    if isinstance(response_payload.get("output"), dict):
        return [response_payload["output"]]
    if isinstance(response_payload.get("output"), list):
        return response_payload["output"]
    if isinstance(response_payload.get("choices"), list):
        messages: list[Any] = []
        for choice in response_payload["choices"]:
            if isinstance(choice, dict):
                messages.append(choice.get("message"))
        return messages
    return []


def _primary_assistant_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() == "assistant":
            return message
    return messages[-1]


def _json_or_str(payload: Any) -> str:
    try:
        return json.dumps(payload, default=str, ensure_ascii=True)
    except Exception:  # noqa: BLE001
        return str(payload)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str, ensure_ascii=True)
        return value
    except Exception:  # noqa: BLE001
        return str(value)


def _message_finish_reason(message: dict[str, Any]) -> str | None:
    response_meta = message.get("response_metadata")
    if isinstance(response_meta, dict):
        finish = response_meta.get("finish_reason") or response_meta.get("done_reason") or response_meta.get(
            "stop_reason"
        )
        if finish is not None:
            return str(finish)
    additional_kwargs = message.get("additional_kwargs")
    if isinstance(additional_kwargs, dict):
        finish = additional_kwargs.get("finish_reason") or additional_kwargs.get("stop_reason")
        if finish is not None:
            return str(finish)
    return None


def _response_model(payload: Any, messages: list[dict[str, Any]]) -> str | None:
    for candidate in _response_metadata_candidates(payload, messages):
        model = candidate.get("model") or candidate.get("model_name")
        if model is not None:
            text = str(model).strip()
            if text:
                return text
    return None


def _response_metadata(payload: Any, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        raw = payload.get("response_metadata")
        if isinstance(raw, dict) and raw:
            return raw
    else:
        raw = getattr(payload, "response_metadata", None)
        if isinstance(raw, dict) and raw:
            return raw

    for message in messages:
        response_meta = message.get("response_metadata")
        if isinstance(response_meta, dict) and response_meta:
            return response_meta

    llm_output = _llm_output(payload)
    if isinstance(llm_output, dict) and llm_output:
        return llm_output

    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata:
            return metadata
    else:
        metadata = getattr(payload, "metadata", None)
        if isinstance(metadata, dict) and metadata:
            return metadata
    return None


def _response_usage(payload: Any, messages: list[dict[str, Any]]) -> dict[str, int] | None:
    # Prefer explicit usage blobs, then infer from response metadata counters.
    usage_candidates: list[Any] = []
    if isinstance(payload, dict):
        usage_candidates.extend((payload.get("usage"), payload.get("token_usage"), payload.get("usage_metadata")))
    else:
        usage_candidates.extend((getattr(payload, "usage", None), getattr(payload, "usage_metadata", None)))

    llm_output = _llm_output(payload)
    if llm_output is not None:
        usage_candidates.extend((llm_output.get("token_usage"), llm_output.get("usage"), llm_output))

    for message in messages:
        usage_candidates.extend((message.get("usage"), message.get("usage_metadata")))
        response_meta = message.get("response_metadata")
        if isinstance(response_meta, dict):
            usage_candidates.extend((response_meta.get("token_usage"), response_meta.get("usage"), response_meta))

    for candidate in usage_candidates:
        normalized = _normalize_usage(candidate)
        if normalized:
            return normalized
    return None


def _response_metadata_candidates(payload: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("response_metadata", "metadata"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        llm_output = payload.get("llm_output")
        if isinstance(llm_output, dict):
            candidates.append(llm_output)
    else:
        for key in ("response_metadata", "metadata"):
            value = getattr(payload, key, None)
            if isinstance(value, dict):
                candidates.append(value)
        llm_output = _llm_output(payload)
        if isinstance(llm_output, dict):
            candidates.append(llm_output)

    for message in messages:
        response_meta = message.get("response_metadata")
        if isinstance(response_meta, dict):
            candidates.append(response_meta)
    return candidates


def _llm_output(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        llm_output = payload.get("llm_output")
    else:
        llm_output = getattr(payload, "llm_output", None)
    if isinstance(llm_output, dict):
        return llm_output
    return None


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None

    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_eval_count")
        or usage.get("input_token_count")
    )
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("eval_count")
        or usage.get("output_token_count")
    )
    total_tokens = usage.get("total_tokens") or usage.get("token_count")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    normalized: dict[str, int] = {}
    if prompt_tokens is not None:
        normalized["prompt_tokens"] = int(prompt_tokens)
    if completion_tokens is not None:
        normalized["completion_tokens"] = int(completion_tokens)
    if total_tokens is not None:
        normalized["total_tokens"] = int(total_tokens)
    return normalized or None


def _usage_from_tokens(token_usage: dict[str, int]) -> dict[str, int] | None:
    normalized = _normalize_usage(token_usage)
    if not normalized:
        return None
    usage: dict[str, int] = {}
    if "prompt_tokens" in normalized:
        usage["prompt_tokens"] = normalized["prompt_tokens"]
    if "completion_tokens" in normalized:
        usage["completion_tokens"] = normalized["completion_tokens"]
    if "total_tokens" in normalized:
        usage["total_tokens"] = normalized["total_tokens"]
    return usage or None

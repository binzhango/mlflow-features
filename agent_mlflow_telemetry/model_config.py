"""Helpers for extracting UI-friendly model configuration attributes."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Mapping

MODEL_CONFIG_KEYS = (
    "temperature",
    "reasoning",
    "reasoning_effort",
    "top_p",
    "top_k",
    "max_tokens",
    "max_output_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "n",
    "stop",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
)


def extract_model_config(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract selected model knobs from one or more config-like dictionaries."""

    config: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in MODEL_CONFIG_KEYS:
            value = source.get(key)
            if value is None:
                continue
            config[key] = _json_safe(value)

    _normalize_reasoning(config)
    return config


def extract_model_config_from_serialized_repr(serialized_repr: str | None) -> dict[str, Any]:
    """Extract model config hints from a serialized model repr string."""

    if not isinstance(serialized_repr, str) or not serialized_repr.strip():
        return {}

    config: dict[str, Any] = {}

    reasoning_match = re.search(
        r"reasoning\s*=\s*\{[^}]*['\"](?:effort|level)['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        serialized_repr,
    )
    if reasoning_match:
        config["reasoning_effort"] = reasoning_match.group(1).strip()

    scalar_keys = (
        "temperature",
        "reasoning",
        "reasoning_effort",
        "top_p",
        "top_k",
        "max_tokens",
        "max_output_tokens",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "n",
    )
    for key in scalar_keys:
        raw = _capture_assignment_value(serialized_repr, key)
        if raw is None:
            continue
        value = _parse_literal(raw)
        if value is not None:
            config[key] = _json_safe(value)

    _normalize_reasoning(config)
    return config


def _normalize_reasoning(config: dict[str, Any]) -> None:
    reasoning = config.get("reasoning")
    if isinstance(reasoning, Mapping):
        effort = reasoning.get("effort") or reasoning.get("level")
        if effort is not None:
            config.setdefault("reasoning_effort", str(effort))
    elif isinstance(reasoning, str) and reasoning.strip():
        config.setdefault("reasoning_effort", reasoning.strip())

    if "reasoning_effort" in config:
        config["reasoning_effort"] = str(config["reasoning_effort"]).strip()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str, ensure_ascii=True)
        return value
    except Exception:  # noqa: BLE001
        return str(value)


def _capture_assignment_value(text: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}\s*=\s*([^,\)]+)", text)
    if not match:
        return None
    return match.group(1).strip()


def _parse_literal(raw: str) -> Any | None:
    value = raw.strip()
    if not value:
        return None

    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]

    try:
        return ast.literal_eval(value)
    except Exception:  # noqa: BLE001
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return value

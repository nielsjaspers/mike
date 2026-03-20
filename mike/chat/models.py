"""Supported model registry for Mike."""

from __future__ import annotations

from typing import Any

DEFAULT_ANTHROPIC_MAX_TOKENS = 127000

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "kimi-k2.5": {
        "vision": True,
        "description": "Moonshot Kimi K2.5 - best overall with vision",
        "api_type": "openai-compatible",
        "max_output_tokens": 260000,
        "endpoint": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "reasoning_param": "reasoning_effort",
        "reasoning_value": "high",
    },
    "glm-5": {
        "vision": False,
        "description": "Zhipu GLM-5 - fast text-only model",
        "api_type": "openai-compatible",
        "max_output_tokens": 202000,
        "endpoint": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "reasoning_param": "reasoning_effort",
        "reasoning_value": "high",
    },
    "minimax-m2.5": {
        "vision": False,
        "description": "MiniMax M2.5 - anthropic-compatible reasoning model",
        "api_type": "anthropic-compatible",
        "max_output_tokens": 196000,
        "endpoint": "/messages",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "reasoning_param": "thinking",
        "reasoning_value": {"type": "enabled"},
    },
    "minimax-m2.7": {
        "vision": False,
        "description": "MiniMax M2.7 - anthropic-compatible reasoning model",
        "api_type": "anthropic-compatible",
        "max_output_tokens": 196000,
        "endpoint": "/messages",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "reasoning_param": "thinking",
        "reasoning_value": {"type": "enabled"},
    },
}

DEFAULT_MODEL = "kimi-k2.5"


def get_model(model_id: str) -> dict[str, Any] | None:
    return SUPPORTED_MODELS.get(model_id)


def model_supports_vision(model_id: str) -> bool:
    return bool((get_model(model_id) or {}).get("vision"))


def clamp_max_tokens(model_id: str, requested: int) -> int | None:
    cfg = get_model(model_id) or {}
    api_type = str(cfg.get("api_type") or "")
    max_output = cfg.get("max_output_tokens")
    limit = int(max_output) if max_output else DEFAULT_ANTHROPIC_MAX_TOKENS
    if api_type == "openai-compatible":
        if requested <= 0 or requested == DEFAULT_ANTHROPIC_MAX_TOKENS:
            return None
        return max(1, min(requested, limit))
    if requested <= 0 or requested == DEFAULT_ANTHROPIC_MAX_TOKENS:
        return limit
    return max(1, min(requested, limit))

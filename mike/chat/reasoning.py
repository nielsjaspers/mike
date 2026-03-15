"""Reasoning policy helpers for Mike."""

from __future__ import annotations

from typing import Any

from mike.chat.models import get_model


def build_reasoning_kwargs(
    model_id: str, minimax_budget_tokens: int | None = None
) -> dict[str, Any]:
    cfg = get_model(model_id) or {}
    if cfg.get("api_type") == "anthropic-compatible":
        value = dict(cfg.get("reasoning_value") or {"type": "enabled"})
        if minimax_budget_tokens:
            value["budget_tokens"] = minimax_budget_tokens
        return {"thinking": value, "reasoning_effort": None}
    return {
        "thinking": None,
        "reasoning_effort": cfg.get("reasoning_value", "high"),
    }

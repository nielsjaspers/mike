"""Provider wrapper for Mike."""

from __future__ import annotations

from mike.chat.models import DEFAULT_MODEL
from mike.config import MikeConfig
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.providers.custom_provider import CustomProvider


def make_provider(config: MikeConfig) -> LLMProvider:
    provider = CustomProvider(
        api_key=config.opencode_api_key or "no-key",
        api_base=config.opencode_api_base,
        default_model=config.default_model or DEFAULT_MODEL,
    )
    provider.generation = GenerationSettings(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        reasoning_effort=None,
    )
    return provider

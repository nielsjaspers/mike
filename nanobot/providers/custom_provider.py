"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import json_repair
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):
    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        # Keep affinity stable for this provider instance to improve backend cache locality.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._prepare_messages(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        reasoning_content = self._extract_reasoning_content(msg)
        tool_calls = [
            ToolCallRequest(
                id=self._normalize_tool_call_id(tc.id),
                name=tc.function.name,
                arguments=json_repair.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments,
            )
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            }
            if u
            else {},
            reasoning_content=reasoning_content,
        )

    def get_default_model(self) -> str:
        return self.default_model

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = self._sanitize_empty_content(messages)
        id_map: dict[str, str] = {}
        result: list[dict[str, Any]] = []
        for msg in prepared:
            clean = dict(msg)
            if clean.get("role") == "assistant" and clean.get("tool_calls"):
                clean["tool_calls"] = [
                    self._prepare_tool_call(tc, id_map)
                    for tc in clean["tool_calls"]
                    if isinstance(tc, dict)
                ]
                if "reasoning_content" not in clean:
                    clean["reasoning_content"] = ""
            if clean.get("role") == "tool" and isinstance(clean.get("tool_call_id"), str):
                clean["tool_call_id"] = self._normalize_tool_call_id(clean["tool_call_id"])
            result.append(clean)
        return result

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _prepare_tool_call(
        self, tool_call: dict[str, Any], id_map: dict[str, str]
    ) -> dict[str, Any]:
        clean = dict(tool_call)
        tool_id = clean.get("id")
        if isinstance(tool_id, str):
            normalized = id_map.setdefault(tool_id, self._normalize_tool_call_id(tool_id))
            clean["id"] = normalized
        function = clean.get("function")
        if isinstance(function, dict):
            clean["function"] = dict(function)
        return clean

    @staticmethod
    def _extract_reasoning_content(message: Any) -> str | None:
        direct = getattr(message, "reasoning_content", None)
        if isinstance(direct, str) and direct:
            return direct
        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        details = getattr(message, "reasoning_details", None)
        if isinstance(details, list):
            texts = []
            for item in details:
                text = (
                    getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
                )
                if isinstance(text, str) and text:
                    texts.append(text)
            if texts:
                return "\n".join(texts)
        return None

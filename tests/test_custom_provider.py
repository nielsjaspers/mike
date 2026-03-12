from types import SimpleNamespace

from nanobot.providers.custom_provider import CustomProvider


def test_prepare_messages_adds_reasoning_content_for_assistant_tool_turn() -> None:
    provider = CustomProvider(default_model="kimi-k2.5")

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "save_memory:0",
                    "type": "function",
                    "function": {"name": "save_memory", "arguments": '{"x":1}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "save_memory:0",
            "name": "save_memory",
            "content": "ok",
        },
    ]

    prepared = provider._prepare_messages(messages)

    assert prepared[0]["reasoning_content"] == ""
    assert prepared[0]["tool_calls"][0]["id"] == prepared[1]["tool_call_id"]
    assert len(prepared[0]["tool_calls"][0]["id"]) == 9


def test_parse_uses_reasoning_field_when_reasoning_content_missing() -> None:
    provider = CustomProvider(default_model="kimi-k2.5")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="save_memory:0",
                            function=SimpleNamespace(
                                name="save_memory",
                                arguments='{"history_entry":"x","memory_update":"y"}',
                            ),
                        )
                    ],
                    reasoning="hidden reasoning",
                    reasoning_details=None,
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    parsed = provider._parse(response)

    assert parsed.reasoning_content == "hidden reasoning"
    assert len(parsed.tool_calls[0].id) == 9
    assert parsed.tool_calls[0].id.isalnum()

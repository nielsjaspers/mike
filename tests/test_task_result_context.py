from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage


@pytest.mark.asyncio
async def test_system_task_result_is_recorded_without_requerying_llm(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SubagentManager"),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)

    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
    loop._run_agent_loop = AsyncMock()

    msg = InboundMessage(
        channel="system",
        sender_id="subagent",
        chat_id="telegram:123",
        content="OpenCode task result for 'research' (completed).\n\nActual result:\nreal findings",
        metadata={
            "_task_result": True,
            "task_id": "abc123",
            "task_label": "research",
            "task_status": "completed",
        },
    )

    result = await loop._process_message(msg)

    assert result is None
    loop._run_agent_loop.assert_not_called()
    session = loop.sessions.get_or_create("telegram:123")
    assert session.messages[-1]["role"] == "system"
    assert session.messages[-1]["task_id"] == "abc123"
    assert "real findings" in session.messages[-1]["content"]

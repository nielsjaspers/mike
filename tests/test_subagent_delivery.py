import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import RunningTaskInfo, SubagentManager
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus


@pytest.mark.asyncio
async def test_announce_result_sends_direct_outbound_message(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    bus = MessageBus()
    bus.publish_outbound = AsyncMock()
    bus.publish_inbound = AsyncMock()
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)

    current = asyncio.current_task()
    assert current is not None
    mgr._running_tasks["task1"] = RunningTaskInfo(
        task_id="task1",
        label="research",
        backend="opencode",
        raw_task=current,
        session_key="telegram:123",
        task="research async queues",
    )

    await mgr._announce_result(
        "task1",
        "research",
        "research async queues",
        "Here are the actual findings.",
        {"channel": "telegram", "chat_id": "123"},
        "ok",
    )

    bus.publish_outbound.assert_awaited_once()
    bus.publish_inbound.assert_awaited_once()
    outbound_call = bus.publish_outbound.await_args
    assert outbound_call is not None
    outbound = outbound_call.args[0]
    assert isinstance(outbound, OutboundMessage)
    assert outbound.channel == "telegram"
    assert outbound.chat_id == "123"
    assert "Here are the actual findings." in outbound.content
    assert outbound.metadata["task_id"] == "task1"
    assert outbound.metadata["task_text"] == "research async queues"

    inbound_call = bus.publish_inbound.await_args
    assert inbound_call is not None
    inbound = inbound_call.args[0]
    assert inbound.channel == "system"
    assert inbound.chat_id == "telegram:123"
    assert "Actual result" in inbound.content
    assert inbound.metadata["task_id"] == "task1"


@pytest.mark.asyncio
async def test_spawn_returns_error_when_opencode_disabled(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    result = await mgr.spawn("do thing", use_opencode=True)

    assert result == "Error: OpenCode Serve is not enabled in config."

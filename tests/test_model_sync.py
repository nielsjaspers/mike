from unittest.mock import MagicMock, patch

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig


async def _make_loop(tmp_path):
    provider = MagicMock()
    provider.get_default_model.return_value = "kimi-k2.5"
    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SubagentManager"),
        patch("nanobot.agent.loop.ResearchManager") as MockResearch,
    ):
        MockResearch.return_value.bind_tools.return_value = None
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            opencode_config=OpenCodeServeConfig(enabled=True, model_id="kimi-k2.5"),
        )
    return loop


import pytest


@pytest.mark.asyncio
async def test_model_command_updates_opencode_model(tmp_path) -> None:
    loop = await _make_loop(tmp_path)
    session = loop.sessions.get_or_create("telegram:1")
    msg = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="1", content="/model minimax-m2.5"
    )

    await loop._handle_model_command(msg, session)

    assert session.current_model == "minimax-m2.5"
    assert loop.opencode_config.model_id == "minimax-m2.5"

from unittest.mock import MagicMock, patch

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig


def test_loop_uses_opencode_web_search_when_enabled(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SessionManager"),
        patch("nanobot.agent.loop.SubagentManager") as subagents,
    ):
        subagents.return_value.opencode_enabled = True
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            opencode_config=OpenCodeServeConfig(enabled=True),
        )

    assert loop.tools.has("web_search")
    assert loop.tools.has("opencode_delegate")


def test_loop_registers_opencode_web_search_even_when_disabled(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SessionManager"),
        patch("nanobot.agent.loop.SubagentManager") as subagents,
    ):
        subagents.return_value.opencode_enabled = False
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            opencode_config=OpenCodeServeConfig(enabled=False),
        )

    assert loop.tools.get("web_search").__class__.__name__ == "OpenCodeWebSearchTool"

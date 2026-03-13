from unittest.mock import MagicMock, patch

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig


def test_loop_registers_beaker_style_web_search_and_research(tmp_path) -> None:
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
    assert loop.tools.has("research")
    assert loop.tools.get("web_search").__class__.__name__ == "WebSearchTool"
    assert loop.tools.get("research").__class__.__name__ == "ResearchTool"

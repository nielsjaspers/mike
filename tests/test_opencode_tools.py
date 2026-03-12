from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.opencode import OpenCodeDelegateTool, OpenCodeWebSearchTool


@pytest.mark.asyncio
async def test_opencode_web_search_delegates_to_subagent() -> None:
    manager = MagicMock()
    manager.run_opencode_query = AsyncMock(return_value="started")
    tool = OpenCodeWebSearchTool(manager)
    tool.set_context("telegram", "123")

    result = await tool.execute(query="best sqlite gui", count=3)

    assert result == "started"
    manager.run_opencode_query.assert_awaited_once()
    _, kwargs = manager.run_opencode_query.await_args
    assert kwargs["origin_channel"] == "telegram"
    assert kwargs["origin_chat_id"] == "123"
    assert "best sqlite gui" in kwargs["text"]


@pytest.mark.asyncio
async def test_opencode_delegate_tool_forces_opencode_runtime() -> None:
    manager = MagicMock()
    manager.run_opencode_query = AsyncMock(return_value="delegated")
    tool = OpenCodeDelegateTool(manager)
    tool.set_context("telegram", "123")

    result = await tool.execute(task="Research three RSS readers", label="rss")

    assert result == "delegated"
    _, kwargs = manager.run_opencode_query.await_args
    assert kwargs["label"] == "rss"
    assert "OpenCode Serve harness" in kwargs["text"]

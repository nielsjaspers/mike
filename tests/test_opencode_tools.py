from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.opencode import OpenCodeDelegateTool


@pytest.mark.asyncio
async def test_opencode_delegate_tool_runs_local_task() -> None:
    manager = MagicMock()
    manager.run_local_task = AsyncMock(return_value="delegated")
    tool = OpenCodeDelegateTool(manager)
    tool.set_context("telegram", "123", "minimax-m2.5")

    result = await tool.execute(task="Research three RSS readers", label="rss")

    assert result == "delegated"
    _, kwargs = manager.run_local_task.await_args
    assert kwargs["label"] == "rss"
    assert kwargs["session_key"] == "telegram:123"
    assert kwargs["model"] == "minimax-m2.5"

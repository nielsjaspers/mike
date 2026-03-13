from unittest.mock import AsyncMock, patch

import pytest

from nanobot.agent.tools.web import WebSearchTool


@pytest.mark.asyncio
async def test_web_search_delegates_to_opencode_search() -> None:
    with patch("nanobot.agent.tools.web.OpencodeSearchTool") as MockTool:
        inst = MockTool.return_value
        inst.execute = AsyncMock(return_value='{"results": []}')
        tool = WebSearchTool()
        tool.set_context("telegram", "123", "glm-5")

        result = await tool.execute(query="nanobot")

        assert result == '{"results": []}'
        inst.set_context.assert_called_once_with("telegram", "123", "glm-5")
        inst.execute.assert_awaited_once()

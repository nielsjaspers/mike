from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.opencode_search import OpencodeSearchTool


@pytest.mark.asyncio
async def test_opencode_search_extracts_json(monkeypatch) -> None:
    tool = OpencodeSearchTool(cli_bin="opencode")
    tool.set_context("telegram", "123", "minimax-m2.5")
    monkeypatch.setattr(
        tool,
        "_run",
        AsyncMock(
            return_value=('{"results": [{"title": "A", "url": "u", "snippet": "s"}]}', "", 0)
        ),
    )

    result = await tool.execute(query="karpathy", count=3)

    assert '"results"' in result


@pytest.mark.asyncio
async def test_opencode_search_prefixes_provider_in_model_flag(monkeypatch) -> None:
    tool = OpencodeSearchTool(cli_bin="opencode", provider_id="opencode-go")
    tool.set_context("telegram", "123", "minimax-m2.5")
    seen = {}

    async def fake_run(cmd):
        seen["cmd"] = cmd
        return '{"results": []}', "", 0

    monkeypatch.setattr(tool, "_run", fake_run)

    await tool.execute(query="karpathy")

    assert "--model" in seen["cmd"]
    idx = seen["cmd"].index("--model")
    assert seen["cmd"][idx + 1] == "opencode-go/minimax-m2.5"


@pytest.mark.asyncio
async def test_opencode_search_rejects_non_local_attach_url() -> None:
    tool = OpencodeSearchTool(cli_bin="opencode")

    result = await tool.execute(query="karpathy", attach_url="https://example.com")

    assert result == "Error: attach_url must be localhost (privacy policy)"

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig, ResearchConfig
from nanobot.research.manager import ResearchManager
from nanobot.research.models import ResearchTask


@pytest.mark.asyncio
async def test_inject_context_persists_and_forwards(tmp_path, monkeypatch) -> None:
    mgr = ResearchManager(
        workspace=tmp_path,
        provider=MagicMock(),
        bus=MessageBus(),
        tools=MagicMock(),
        model="kimi-k2.5",
        config=ResearchConfig(),
        opencode_config=OpenCodeServeConfig(enabled=True),
    )
    task = ResearchTask(
        task_id="abc123",
        session_key="telegram:1",
        origin_channel="telegram",
        origin_chat_id="1",
        query="query",
        backend="opencode",
        status="running",
        backend_session_id="sess-1",
    )
    mgr.persistence.save_task(task)
    fake = AsyncMock()
    fake.prompt.return_value = {}
    fake.aclose.return_value = None
    monkeypatch.setattr(mgr, "_client", lambda model: fake)

    result = await mgr.inject_context("telegram:1", "extra context")
    updated = mgr.persistence.load_task("abc123")

    assert result == "Context added to research task abc123."
    assert updated is not None
    assert updated.user_injections[-1] == "extra context"
    fake.prompt.assert_awaited_once()

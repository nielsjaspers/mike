from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig, ResearchConfig
from nanobot.research.manager import ResearchManager
from nanobot.research.models import ResearchTask


@pytest.mark.asyncio
async def test_resume_pending_starts_running_tasks(tmp_path, monkeypatch) -> None:
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
        status="running",
        backend="opencode",
    )
    mgr.persistence.save_task(task)
    monkeypatch.setattr(mgr, "_run_task", AsyncMock())

    await mgr.resume_pending()

    assert "abc123" in mgr._running

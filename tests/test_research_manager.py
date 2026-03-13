from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig, ResearchConfig
from nanobot.research.manager import ResearchManager


@pytest.mark.asyncio
async def test_start_task_persists_opencode_research(tmp_path) -> None:
    mgr = ResearchManager(
        workspace=tmp_path,
        provider=MagicMock(),
        bus=MessageBus(),
        tools=MagicMock(),
        model="kimi-k2.5",
        config=ResearchConfig(),
        opencode_config=OpenCodeServeConfig(enabled=True),
    )

    msg = await mgr.start_task(
        query="research karpathy autoresearch",
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        model="minimax-m2.5",
    )

    task = mgr.list_tasks("telegram:1")[0]
    assert "started via opencode runtime" in msg
    assert task.backend == "opencode"
    assert task.model == "minimax-m2.5"


@pytest.mark.asyncio
async def test_run_local_task_for_delegate_uses_same_manager(tmp_path) -> None:
    mgr = ResearchManager(
        workspace=tmp_path,
        provider=MagicMock(),
        bus=MessageBus(),
        tools=MagicMock(),
        model="kimi-k2.5",
        config=ResearchConfig(),
        opencode_config=OpenCodeServeConfig(enabled=True),
    )

    msg = await mgr.run_local_task(
        task="build a poc",
        label="poc",
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        model="glm-5",
    )

    task = mgr.list_tasks("telegram:1")[0]
    assert "started via opencode runtime" in msg
    assert task.title == "poc"
    assert task.model == "glm-5"

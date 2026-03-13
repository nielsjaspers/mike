from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import RunningTaskInfo, SubagentManager
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig
from nanobot.opencode_client import OpencodeServeClient


@pytest.mark.asyncio
async def test_inject_context_returns_message_when_no_running_task(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    result = await mgr.inject_context("telegram:123", "more info")

    assert result == "No running OpenCode task found for this chat."


@pytest.mark.asyncio
async def test_spawn_uses_explicit_opencode_backend(tmp_path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    config = OpenCodeServeConfig(enabled=True)
    mgr = SubagentManager(
        provider=provider, workspace=tmp_path, bus=MessageBus(), opencode_config=config
    )

    started = asyncio.Event()

    async def fake_run(task_id: str, task: str, label: str, origin: dict[str, str]) -> None:
        started.set()

    monkeypatch.setattr(mgr, "_run_opencode_task", fake_run)

    result = await mgr.spawn("investigate this", use_opencode=True)
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert "via opencode runtime" in result


@pytest.mark.asyncio
async def test_opencode_task_waits_for_final_message(tmp_path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    responses = [[], [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "done"}]}]]

    async def fake_list_messages(self, session_id: str):
        assert session_id == "sess-1"
        return responses.pop(0)

    monkeypatch.setattr(OpencodeServeClient, "list_messages", fake_list_messages)
    monkeypatch.setattr("nanobot.agent.subagent.asyncio.sleep", AsyncMock(return_value=None))

    client = MagicMock()
    client.list_messages = AsyncMock(
        side_effect=[
            [],
            [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "done"}]}],
        ]
    )

    result = await mgr._wait_for_opencode_result(client, "sess-1")

    assert result == "done"


@pytest.mark.asyncio
async def test_opencode_task_waits_past_tool_call_assistant_messages(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())
    original_sleep = asyncio.sleep

    client = MagicMock()
    client.list_messages = AsyncMock(
        side_effect=[
            [
                {
                    "info": {"role": "assistant", "finish": "tool-calls", "time": {"completed": 1}},
                    "parts": [{"type": "reasoning", "text": "searching"}],
                }
            ],
            [
                {
                    "info": {"role": "assistant", "finish": "tool-calls", "time": {"completed": 1}},
                    "parts": [{"type": "reasoning", "text": "searching"}],
                },
                {
                    "info": {"role": "assistant", "finish": "stop", "time": {"completed": 2}},
                    "parts": [{"type": "text", "text": "final answer"}],
                },
            ],
        ]
    )

    async def fake_sleep(_: float) -> None:
        await original_sleep(0)

    with pytest.MonkeyPatch.context() as m:
        m.setattr("nanobot.agent.subagent.asyncio.sleep", fake_sleep)
        result = await mgr._wait_for_opencode_result(client, "sess-1")

    assert result == "final answer"


@pytest.mark.asyncio
async def test_opencode_task_rejects_reasoning_only_completion(tmp_path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())
    original_sleep = asyncio.sleep

    client = MagicMock()
    client.list_messages = AsyncMock(
        return_value=[
            {
                "info": {"role": "assistant", "time": {"completed": 123}},
                "parts": [{"type": "reasoning", "text": "I should search"}],
            }
        ]
    )

    async def fake_sleep(_: float) -> None:
        await original_sleep(0)

    with pytest.MonkeyPatch.context() as m:
        m.setattr("nanobot.agent.subagent.asyncio.sleep", fake_sleep)
        with pytest.raises(RuntimeError, match="completed without a text result"):
            await mgr._wait_for_opencode_result(client, "sess-1")


@pytest.mark.asyncio
async def test_run_opencode_task_announces_result(tmp_path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    fake_client = AsyncMock()
    fake_client.create_session.return_value = {"id": "sess-1"}
    fake_client.prompt_async.return_value = {}
    fake_client.aclose.return_value = None
    monkeypatch.setattr("nanobot.agent.subagent.OpencodeServeClient", lambda **kwargs: fake_client)

    announced: list[tuple[str, str, str, str, dict[str, str], str]] = []

    async def fake_announce(task_id, label, task, result, origin, status):
        announced.append((task_id, label, task, result, origin, status))

    monkeypatch.setattr(mgr, "_wait_for_opencode_result", AsyncMock(return_value="finished"))
    monkeypatch.setattr(mgr, "_announce_result", fake_announce)

    current = asyncio.current_task()
    assert current is not None
    mgr._running_tasks["task1"] = RunningTaskInfo(
        task_id="task1",
        label="research",
        backend="opencode",
        raw_task=current,
        task="research this",
    )

    await mgr._run_opencode_task(
        "task1", "research this", "research", {"channel": "telegram", "chat_id": "123"}
    )

    assert mgr._running_tasks["task1"].status == "completed"
    fake_client.create_session.assert_awaited_once()
    fake_client.prompt_async.assert_awaited_once()
    assert announced == [
        (
            "task1",
            "research",
            "research this",
            "finished",
            {"channel": "telegram", "chat_id": "123"},
            "ok",
        )
    ]


@pytest.mark.asyncio
async def test_native_iteration_limit_comes_from_config(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(
        provider=provider, workspace=tmp_path, bus=MessageBus(), native_max_iterations=22
    )

    assert mgr.native_max_iterations == 22


def test_native_subagent_tools_include_web_search(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    tools = mgr._build_native_tools()

    assert tools.has("web_search")

from __future__ import annotations

import re

import pytest

from mike.agent.loop import AgentLoop
from mike.bus import MessageBus
from mike.config import MikeConfig
from mike.llm import LLMProvider, LLMResponse, ToolCallRequest
from mike.storage.chats import ChatStore
from mike.storage.tasks import TaskStore
from mike.tasks.manager import TaskManager
from mike.tasks.research import ResearchManager
from mike.types import InboundMessage


class FakeProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
        thinking=None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice, thinking
        self._call_count += 1

        last_msg = messages[-1] if messages else {}
        is_tool_result = last_msg.get("role") == "tool"

        if is_tool_result:
            tool_content = last_msg.get("content", "")
            if isinstance(tool_content, str):
                import json

                try:
                    data = json.loads(tool_content)
                    action = data.get("action", "")
                    if action == "create":
                        msg = data.get("message", "")
                        return LLMResponse(content=f"OK. {msg}")
                    if action == "list":
                        items = data.get("data", {}).get("items", [])
                        if not items:
                            return LLMResponse(content="No schedules found.")
                        lines = [f"{item['id']}: {item['title']}" for item in items]
                        return LLMResponse(content="Schedules:\n" + "\n".join(lines))
                    if action == "delete":
                        return LLMResponse(
                            content=f"Deleted schedule {data.get('data', {}).get('schedule_id', '')}."
                        )
                    if action == "help":
                        return LLMResponse(content=data.get("message", ""))
                except (json.JSONDecodeError, ValueError):
                    pass
            return LLMResponse(content="Done.")

        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                user_content = msg["content"]
                break

        action, args = self._parse_action(user_content)
        if action == "help":
            return LLMResponse(
                content="/schedule - Manage schedules. Use: /schedule add <time>, <prompt>, /schedule list, /schedule show <id>, /schedule delete <id>"
            )
        if action == "list":
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="schedule", arguments={"action": "list"})
                ],
            )
        if action == "create":
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="schedule",
                        arguments={
                            "action": "create",
                            "text": args.get("text", ""),
                            "kind": "reminder",
                        },
                    )
                ],
            )
        if action == "delete":
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="schedule",
                        arguments={"action": "delete", "schedule_id": args.get("schedule_id")},
                    )
                ],
            )
        return LLMResponse(content="Unknown command.")

    def _parse_action(self, content: str) -> tuple[str, dict]:
        if not content:
            return "help", {}
        text = content.strip()
        last_line = text.split("\n")[-1].strip() if "\n" in text else text

        delete_match = re.search(r"delete\s+(sc_\w+)", last_line)
        if delete_match:
            return "delete", {"schedule_id": delete_match.group(1)}

        if last_line.startswith("/schedule delete") or last_line.startswith("/schedule del"):
            return "delete", {}

        if re.match(r"list\s*$", last_line, re.IGNORECASE):
            return "list", {}

        add_match = re.search(r"add\s+(.+)", last_line, re.IGNORECASE)
        if add_match:
            return "create", {"text": add_match.group(1).strip()}

        if re.search(r"remind", last_line, re.IGNORECASE):
            return "create", {"text": last_line}

        if last_line.strip() in ("/schedule", "/schedule help", "help"):
            return "help", {}

        if re.match(r"list\s*$", last_line, re.IGNORECASE):
            return "list", {}

        return "help", {}

    def get_default_model(self) -> str:
        return "kimi-k2.5"


def make_loop(tmp_path, provider):
    cfg = MikeConfig(data_dir=str(tmp_path / "mike-data"), project_root=str(tmp_path))
    bus = MessageBus()
    store = ChatStore(cfg)
    tasks = TaskStore(cfg.data_dir_path / "tasks")
    manager = TaskManager(tasks)
    research = ResearchManager(cfg, bus, tasks, manager)
    loop = AgentLoop(bus=bus, provider=provider, config=cfg, store=store, research=research)
    return loop, bus, store


def setup_schedule_manager(loop, tmp_path, bus):
    from mike.scheduling.manager import ScheduleManager
    from mike.scheduling.store import ScheduleStore

    cfg = MikeConfig(
        data_dir=str(tmp_path / "mike-data"),
        project_root=str(tmp_path),
        schedule_enabled=True,
    )
    schedule_store = ScheduleStore(tmp_path / "schedules")
    schedule = ScheduleManager(cfg, bus, schedule_store)
    loop.schedule_manager = schedule
    schedule_tool = loop.tools.get("schedule")
    if schedule_tool:
        schedule_tool.set_manager(schedule)
    return schedule


@pytest.mark.asyncio
async def test_schedule_help_command(tmp_path):
    provider = FakeProvider()
    loop, bus, store = make_loop(tmp_path, provider)
    setup_schedule_manager(loop, tmp_path, bus)

    result = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/schedule")
    )
    assert result is not None
    assert "/schedule" in result.content


@pytest.mark.asyncio
async def test_schedule_list_command(tmp_path):
    provider = FakeProvider()
    loop, bus, store = make_loop(tmp_path, provider)
    setup_schedule_manager(loop, tmp_path, bus)

    result = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/schedule list")
    )
    assert result is not None
    assert "No schedules" in result.content


@pytest.mark.asyncio
async def test_schedule_add_and_list(tmp_path):
    provider = FakeProvider()
    loop, bus, store = make_loop(tmp_path, provider)
    setup_schedule_manager(loop, tmp_path, bus)

    add_result = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u",
            chat_id="direct",
            content="/schedule add in 5 minutes, test reminder",
        )
    )
    assert add_result is not None
    assert "Created schedule" in add_result.content

    list_result = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/schedule list")
    )
    assert list_result is not None
    assert "test reminder" in list_result.content


@pytest.mark.asyncio
async def test_schedule_delete(tmp_path):
    provider = FakeProvider()
    loop, bus, store = make_loop(tmp_path, provider)
    setup_schedule_manager(loop, tmp_path, bus)

    add_result = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u",
            chat_id="direct",
            content="/schedule add in 5 minutes, to delete",
        )
    )
    assert add_result is not None
    schedule_id = add_result.content.split("[")[1].split("]")[0]

    delete_result = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u",
            chat_id="direct",
            content=f"/schedule delete {schedule_id}",
        )
    )
    assert delete_result is not None
    assert "Deleted" in delete_result.content

    list_result = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/schedule list")
    )
    assert list_result is not None
    assert "No schedules" in list_result.content

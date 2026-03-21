from __future__ import annotations

from pathlib import Path

import pytest

from mike.agent.loop import AgentLoop
from mike.bus import MessageBus
from mike.config import MikeConfig
from mike.llm import LLMProvider, LLMResponse
from mike.storage.chats import ChatStore
from mike.storage.tasks import TaskStore
from mike.tasks.manager import TaskManager
from mike.tasks.research import ResearchManager
from mike.types import InboundMessage


class FakeProvider(LLMProvider):
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
        text = "ok"
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content")
                if isinstance(content, str) and content.strip():
                    text = f"temp:{content.strip().splitlines()[-1]}"
        return LLMResponse(content=text)

    def get_default_model(self) -> str:
        return "kimi-k2.5"


@pytest.mark.asyncio
async def test_attach_fork_sessions_and_temp_flow(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path / "mike-data"), project_root=str(tmp_path))
    bus = MessageBus()
    store = ChatStore(cfg)
    tasks = TaskStore(cfg.data_dir_path / "tasks")
    manager = TaskManager(tasks)
    research = ResearchManager(cfg, bus, tasks, manager)
    loop = AgentLoop(bus=bus, provider=FakeProvider(), config=cfg, store=store, research=research)

    first = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="hello")
    )
    assert first is not None

    runtime_key = "cli:direct"
    active_id = store.resolve_active_session(runtime_key)
    active_session = store.get_or_create_session_record(active_id)
    assert active_session.messages

    created = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/new")
    )
    assert created is not None
    assert "saved" in created.content.lower()
    saved_id = active_id

    fresh_id = store.resolve_active_session(runtime_key)
    assert fresh_id != saved_id

    attached = await loop._process_message(
        InboundMessage(
            channel="cli", sender_id="u", chat_id="direct", content=f"/attach {saved_id}"
        )
    )
    assert attached is not None
    assert f"`{saved_id}`" in attached.content

    same_attach = await loop._process_message(
        InboundMessage(
            channel="cli", sender_id="u", chat_id="direct", content=f"/attach {saved_id}"
        )
    )
    assert same_attach is not None
    assert "already in session" in same_attach.content.lower()

    not_found = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/attach zzzzz")
    )
    assert not_found is not None
    assert "not found" in not_found.content.lower()

    forked = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content=f"/fork {saved_id}")
    )
    assert forked is not None
    assert "forked session" in forked.content.lower()

    listed = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/sessions")
    )
    assert listed is not None
    assert "Page 1 of" in listed.content

    searched = await loop._process_message(
        InboundMessage(
            channel="cli", sender_id="u", chat_id="direct", content="/sessions search hello"
        )
    )
    assert searched is not None
    assert "Page 1 of" in searched.content or "No sessions matching" in searched.content

    temp = await loop._process_message(
        InboundMessage(
            channel="cli", sender_id="u", chat_id="direct", content="/temp what time is it"
        )
    )
    assert temp is not None
    assert "temp:" in temp.content

    btw = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/btw quick check")
    )
    assert btw is not None
    assert "temp:" in btw.content


@pytest.mark.asyncio
async def test_sessions_page_validation_and_clear_no_save(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path / "mike-data"), project_root=str(tmp_path))
    bus = MessageBus()
    store = ChatStore(cfg)
    tasks = TaskStore(cfg.data_dir_path / "tasks")
    manager = TaskManager(tasks)
    research = ResearchManager(cfg, bus, tasks, manager)
    loop = AgentLoop(bus=bus, provider=FakeProvider(), config=cfg, store=store, research=research)

    invalid_page = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/sessions nope")
    )
    assert invalid_page is not None
    assert invalid_page.content == "Usage: /sessions [page]"

    empty_search = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/sessions search")
    )
    assert empty_search is not None
    assert empty_search.content == "Usage: /sessions search <query>"

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="hello")
    )
    runtime_key = "cli:direct"
    session_id = store.resolve_active_session(runtime_key)

    clear = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/clear")
    )
    assert clear is not None
    assert clear.content == "Chat cleared."

    assert not store.session_record_exists(session_id)

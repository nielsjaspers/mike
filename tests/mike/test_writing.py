from __future__ import annotations

from pathlib import Path

import pytest

from mike.bus import MessageBus
from mike.config import MikeConfig
from mike.writing.manager import WritingManager
from mike.writing.store import WritingStore
from mike.writing.types import StoryState, WorkMetadata, WritingMetadata


class DummyLoop:
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress=None,
        creative: bool = False,
        model: str | None = None,
    ) -> str:
        del content, session_key, channel, chat_id, on_progress, creative, model
        return "# Test Title\n\nA first sentence. A second sentence."


def test_round_robin_rotation():
    meta = WritingMetadata()
    assert meta.next_piece_type() == "poetry"
    assert meta.next_piece_type() == "philosophy"
    assert meta.next_piece_type() == "tech_speculation"
    assert meta.next_piece_type() == "fiction"
    assert meta.next_piece_type() == "poetry"


def test_story_state_serialization_roundtrip():
    state = StoryState(
        story_id="story-1",
        title="The Last Engine",
        directive="start",
        genre="sci-fi",
        premise="premise",
        chapters_completed=2,
        chapter_summaries=["one", "two"],
    )
    restored = StoryState.from_dict(state.to_dict())
    assert restored.story_id == "story-1"
    assert restored.chapters_completed == 2
    assert restored.chapter_summaries == ["one", "two"]


def test_writing_store_save_load(tmp_path: Path):
    store = WritingStore(tmp_path / "writing")
    meta = WritingMetadata()
    work = WorkMetadata(
        work_id="2026-03-20_poetry_abc123",
        title="Night Shift",
        piece_type="poetry",
        timestamp="2026-03-20T03:00:00",
        summary="summary",
        file_path="works/2026-03-20_poetry_abc123.md",
    )
    store.save_work(work, "# Night Shift\n")
    meta.add_work(work)
    store.save_metadata(meta)
    loaded = store.load_metadata()
    assert loaded.total_pieces == 1
    assert loaded.works[0]["title"] == "Night Shift"
    stories = store.list_stories()
    assert stories == []


def test_title_and_summary_extraction(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path), project_root=str(tmp_path))
    manager = WritingManager(cfg, MessageBus(), WritingStore(tmp_path / "writing"), DummyLoop())
    title = manager._extract_title("# Hello World\n\nBody")
    summary = manager._extract_summary(
        "# Hello\n\nFirst sentence. Second sentence. Third sentence."
    )
    assert title == "Hello World"
    assert summary == "First sentence. Second sentence."


def test_seconds_until_returns_positive_window(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path), project_root=str(tmp_path))
    manager = WritingManager(cfg, MessageBus(), WritingStore(tmp_path / "writing"), DummyLoop())
    value = manager._seconds_until("03:00")
    assert 0 < value <= 24 * 3600


@pytest.mark.asyncio
async def test_execute_session_saves_work(tmp_path: Path):
    cfg = MikeConfig(
        data_dir=str(tmp_path), project_root=str(tmp_path), nocturne_telegram_chat_id=""
    )
    store = WritingStore(tmp_path / "writing")
    manager = WritingManager(cfg, MessageBus(), store, DummyLoop())
    work = await manager._execute_session(
        prompt="Write something",
        session_key="nocturne:test",
        channel="cli",
        chat_id="direct",
        piece_type="poetry",
    )
    assert work.title == "Test Title"
    assert (tmp_path / "writing" / work.file_path).exists()
    loaded = store.load_metadata()
    assert loaded.total_pieces == 1

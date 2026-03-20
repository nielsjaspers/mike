"""Nocturne writing orchestration."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from mike.bus import MessageBus
from mike.common import safe_filename
from mike.config import MikeConfig
from mike.chat.models import SUPPORTED_MODELS
from mike.types import OutboundMessage
from mike.writing.prompts import (
    build_daily_prompt,
    build_story_chapter_prompt,
    build_story_start_prompt,
    load_creative_soul,
)
from mike.writing.store import WritingStore
from mike.writing.types import PIECE_TYPES, StoryState, WorkMetadata


class WritingManager:
    def __init__(self, config: MikeConfig, bus: MessageBus, store: WritingStore, agent_loop: Any):
        self.config = config
        self.bus = bus
        self.store = store
        self.agent_loop = agent_loop
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._session_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.current_task()
        try:
            await self._timer_loop()
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False
            self._task = None

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _timer_loop(self) -> None:
        while self._running:
            seconds = self._seconds_until(self.config.nocturne_time)
            logger.info("Nocturne sleeping for {} seconds", seconds)
            await asyncio.sleep(seconds)
            if not self._running:
                return
            try:
                await self.trigger_daily()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Nocturne daily writing failed")

    def _seconds_until(self, hhmm: str) -> float:
        now = datetime.now()
        try:
            hour_s, minute_s = hhmm.split(":", 1)
            hour = max(0, min(23, int(hour_s)))
            minute = max(0, min(59, int(minute_s)))
        except Exception:
            hour, minute = 3, 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target = target + timedelta(days=1)
        return (target - now).total_seconds()

    async def trigger_daily(self) -> str:
        if self._session_lock.locked():
            logger.warning("Nocturne daily run skipped; writing session already active")
            return "Skipped daily run; another writing session is active."
        target_channel = "telegram" if self.config.nocturne_telegram_chat_id else "cli"
        target_chat_id = self.config.nocturne_telegram_chat_id or "nocturne"
        source_session = (
            f"telegram:{target_chat_id}"
            if target_channel == "telegram" and target_chat_id
            else None
        )
        active_story_id = self._first_active_story_id()
        if active_story_id:
            return await self.continue_story(
                active_story_id,
                session_key=source_session or "nocturne:daily",
                channel=target_channel,
                chat_id=target_chat_id,
            )
        metadata = self.store.load_metadata()
        piece_type = metadata.next_piece_type()
        recent = self.store.list_works(limit=8)
        creative_soul = load_creative_soul(self.config.data_dir_path)
        prompt = build_daily_prompt(piece_type, creative_soul, recent)
        result = await self._execute_session(
            prompt,
            session_key="nocturne:daily",
            channel=target_channel,
            chat_id=target_chat_id,
            piece_type=piece_type,
            model=self._model_for_session(source_session),
        )
        return f"Daily {piece_type} piece written: {result.title}"

    async def write_on_demand(
        self, directive: str, session_key: str, channel: str, chat_id: str
    ) -> str:
        directive = directive.strip()
        if not directive:
            return "Usage: /write <directive>"
        metadata = self.store.load_metadata()
        piece_type = self._infer_piece_type(directive, metadata.next_piece_type())
        recent = self.store.list_works(limit=8)
        creative_soul = load_creative_soul(self.config.data_dir_path)
        prompt = build_daily_prompt(piece_type, creative_soul, recent, directive=directive)
        result = await self._execute_session(
            prompt,
            session_key=f"{session_key}:nocturne:write",
            channel=channel,
            chat_id=chat_id,
            piece_type=piece_type,
            model=self._model_for_session(session_key),
        )
        return f"Saved {piece_type} piece '{result.title}' to {result.file_path}."

    async def start_story(
        self, directive: str, session_key: str, channel: str, chat_id: str
    ) -> str:
        directive = directive.strip()
        if not directive:
            return "Usage: /story start <directive>"
        async with self._session_lock:
            creative_soul = load_creative_soul(self.config.data_dir_path)
            prompt = build_story_start_prompt(directive, creative_soul)
            output = await self.agent_loop.process_direct(
                prompt,
                session_key=f"{session_key}:nocturne:story:{chat_id}",
                channel=channel,
                chat_id=chat_id,
                creative=True,
                model=self._model_for_session(session_key),
            )
        state_payload = self._extract_json_block(output)
        chapter_text = self._remove_json_block(output).strip() or output.strip()
        title = (
            self._state_title(state_payload)
            or self._extract_title(chapter_text)
            or "Untitled Story"
        )
        story_id = self._story_id_from_title(title)
        state = StoryState(
            story_id=story_id,
            title=title,
            directive=directive,
            genre=str(state_payload.get("genre") or "fiction"),
            premise=str(state_payload.get("premise") or directive),
            characters=list(state_payload.get("characters") or []),
            themes=[str(item) for item in (state_payload.get("themes") or [])],
            outline=list(state_payload.get("outline") or []),
            chapter_summaries=[self._extract_summary(chapter_text)],
            chapters_completed=1,
            status="active",
        )
        self.store.save_story(state)
        chapter_path = self.store.save_story_chapter(story_id, 1, chapter_text)
        await self._record_story_work(state, 1, chapter_path, chapter_text, channel, chat_id)
        return f"Story '{title}' started as {story_id}. Chapter 1 saved."

    async def continue_story(
        self, story_id: str, session_key: str, channel: str, chat_id: str
    ) -> str:
        async with self._session_lock:
            story = self.store.load_story(story_id)
            if story is None:
                return f"Story not found: {story_id}"
            last_chapter = self.store.read_story_chapter(story_id, story.chapters_completed)
            creative_soul = load_creative_soul(self.config.data_dir_path)
            prompt = build_story_chapter_prompt(story, creative_soul, last_chapter)
            output = await self.agent_loop.process_direct(
                prompt,
                session_key=f"{session_key}:nocturne:story:{story_id}",
                channel=channel,
                chat_id=chat_id,
                creative=True,
                model=self._model_for_session(session_key),
            )
            next_chapter = story.chapters_completed + 1
            chapter_text = output.strip()
            chapter_path = self.store.save_story_chapter(story_id, next_chapter, chapter_text)
            story.chapters_completed = next_chapter
            story.chapter_summaries.append(self._extract_summary(chapter_text))
            story.updated_at = datetime.now().isoformat()
            self.store.save_story(story)
            await self._record_story_work(
                story, next_chapter, chapter_path, chapter_text, channel, chat_id
            )
            return f"Story '{story.title}' continued. Chapter {next_chapter} saved."

    def format_story_list(self) -> str:
        stories = self.store.list_stories()
        if not stories:
            return "No active stories. Use /story start <directive>."
        lines = ["Stories:"]
        for story in stories[:20]:
            lines.append(
                f"- {story.story_id}: {story.title} [{story.status}] chapters={story.chapters_completed}"
            )
        return "\n".join(lines)

    def _first_active_story_id(self) -> str | None:
        metadata = self.store.load_metadata()
        for story_id in metadata.active_stories:
            story = self.store.load_story(story_id)
            if story and story.status == "active":
                return story.story_id
        return None

    def _model_for_session(self, source_session_key: str | None) -> str | None:
        if source_session_key:
            try:
                chat_session = self.agent_loop.store.get(source_session_key)
                model = (chat_session.current_model or "").strip()
                if model and model in SUPPORTED_MODELS:
                    return model
            except Exception:
                logger.debug("No chat model found for session {}", source_session_key)
        candidate = (self.config.nocturne_model or "").strip()
        if candidate and candidate in SUPPORTED_MODELS:
            return candidate
        return None

    async def _execute_session(
        self,
        prompt: str,
        session_key: str,
        channel: str,
        chat_id: str,
        piece_type: str,
        model: str | None = None,
    ) -> WorkMetadata:
        async with self._session_lock:
            text = await self.agent_loop.process_direct(
                prompt,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                creative=True,
                model=model,
            )
            title = self._extract_title(text)
            summary = self._extract_summary(text)
            now = datetime.now()
            work_id = f"{now:%Y-%m-%d}_{piece_type}_{uuid.uuid4().hex[:6]}"
            filename = safe_filename(f"{work_id}.md")
            work = WorkMetadata(
                work_id=work_id,
                title=title,
                piece_type=piece_type,
                timestamp=now.isoformat(),
                summary=summary,
                file_path=f"works/{filename}",
            )
            self.store.save_work(work, text)
            metadata = self.store.load_metadata()
            if piece_type in PIECE_TYPES:
                metadata.last_piece_type_index = PIECE_TYPES.index(piece_type)
            metadata.add_work(work)
            self.store.save_metadata(metadata)
            await self._notify_saved(work, channel=channel, chat_id=chat_id)
            return work

    async def _record_story_work(
        self,
        story: StoryState,
        chapter: int,
        chapter_path: Path,
        chapter_text: str,
        channel: str,
        chat_id: str,
    ) -> None:
        meta = self.store.load_metadata()
        if story.story_id not in meta.active_stories:
            meta.active_stories.append(story.story_id)
        relative_path = chapter_path.relative_to(self.store.root)
        work = WorkMetadata(
            work_id=f"{story.story_id}-chapter-{chapter:02d}",
            title=f"{story.title} - Chapter {chapter}",
            piece_type="fiction",
            timestamp=datetime.now().isoformat(),
            summary=self._extract_summary(chapter_text),
            file_path=str(relative_path),
            story_id=story.story_id,
            chapter=chapter,
        )
        meta.add_work(work)
        self.store.save_metadata(meta)
        await self._notify_saved(work, channel=channel, chat_id=chat_id)

    async def _notify_saved(
        self, work: WorkMetadata, channel: str | None = None, chat_id: str | None = None
    ) -> None:
        target_channel = channel or "telegram"
        target_chat = chat_id or self.config.nocturne_telegram_chat_id
        if self.config.nocturne_telegram_chat_id:
            target_channel = "telegram"
            target_chat = self.config.nocturne_telegram_chat_id
        if target_channel != "telegram" or not target_chat:
            return
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=target_channel,
                chat_id=target_chat,
                content=(
                    f"New {work.piece_type} piece: '{work.title}'\n"
                    f"{work.summary}\n"
                    f"Saved: {work.file_path}"
                ),
                metadata={"_nocturne": True},
            )
        )

    @staticmethod
    def _infer_piece_type(directive: str, default: str) -> str:
        lower = directive.lower()
        if any(token in lower for token in ("poem", "poetry", "haiku", "sonnet", "verse")):
            return "poetry"
        if any(token in lower for token in ("philosophy", "philosophical", "ethics", "meaning")):
            return "philosophy"
        if any(token in lower for token in ("tech", "technology", "future", "ai", "speculat")):
            return "tech_speculation"
        if any(token in lower for token in ("story", "fiction", "character", "chapter")):
            return "fiction"
        return default if default in PIECE_TYPES else "fiction"

    @staticmethod
    def _extract_title(text: str) -> str:
        for line in text.splitlines():
            match = re.match(r"^#\s+(.+?)\s*$", line.strip())
            if match:
                return match.group(1).strip()
        for line in text.splitlines():
            clean = line.strip()
            if clean:
                return clean[:100]
        return "Untitled"

    @staticmethod
    def _extract_summary(text: str) -> str:
        body = re.sub(r"^#\s+.*$", "", text, flags=re.MULTILINE).strip()
        if not body:
            return "No summary available."
        compact = re.sub(r"\s+", " ", body)
        parts = re.split(r"(?<=[.!?])\s+", compact)
        summary = " ".join(parts[:2]).strip()
        if len(summary) > 220:
            return summary[:217].rstrip() + "..."
        return summary

    @staticmethod
    def _story_id_from_title(title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        slug = slug[:48] or "story"
        return f"{slug}-{uuid.uuid4().hex[:4]}"

    @staticmethod
    def _extract_json_block(text: str) -> dict[str, Any]:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if not match:
            return {}
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _remove_json_block(text: str) -> str:
        return re.sub(r"```json\s*[\s\S]*?\s*```", "", text).strip()

    @staticmethod
    def _state_title(state_payload: dict[str, Any]) -> str:
        title = state_payload.get("title") if isinstance(state_payload, dict) else None
        return str(title).strip() if title else ""

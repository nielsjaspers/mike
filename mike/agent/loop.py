"""Simplified native tool-calling loop for Mike."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from mike.bus import MessageBus
from mike.memory.archive import ArchiveManager
from mike.chat.models import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    clamp_max_tokens,
    model_supports_vision,
)
from mike.chat.prompts import build_system_prompt
from mike.chat.reasoning import build_reasoning_kwargs
from mike.config import MikeConfig
from mike.skills import build_summary
from mike.storage.chats import ChatSession, ChatStore
from mike.tasks.research import ResearchManager
from mike.tools.delegate import OpenCodeDelegateTool
from mike.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from mike.tools.history import GetHistoryConversationTool, SearchHistoryTool
from mike.tools.memory import ReadMemoryTool
from mike.tools.message import MessageTool
from mike.tools.registry import ToolRegistry
from mike.tools.research import ResearchTool
from mike.tools.shell import ExecTool
from mike.tools.web import WebFetchTool, WebSearchTool
from mike.types import InboundMessage, OutboundMessage
from mike.helpers import build_assistant_message, detect_image_mime
from mike.llm import LLMProvider


class ContextBuilder:
    _RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"

    def __init__(self, store: ChatStore):
        self.store = store

    def build_system_prompt(self, session_key: str, creative: bool = False) -> str:
        del session_key
        root = self.store.shared_root
        creative_soul = ""
        if creative:
            creative_path = root / "CREATIVE_SOUL.md"
            if creative_path.exists():
                creative_soul = creative_path.read_text(encoding="utf-8").strip()
        return build_system_prompt(
            root,
            skills_summary=build_summary(root),
            creative_soul=creative_soul,
            creative_mode=creative,
        )

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        lines = [f"Current Time: {now}"]
        if channel and chat_id:
            lines.extend([f"Channel: {channel}", f"Chat ID: {chat_id}"])
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def build_messages(
        self,
        session_key: str,
        history: list[dict[str, Any]],
        current_message: str,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        creative: bool = False,
    ) -> list[dict[str, Any]]:
        runtime = self._build_runtime_context(channel, chat_id)
        user_content = self._build_user_content(current_message, media)
        if isinstance(user_content, str):
            merged = f"{runtime}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime}] + user_content
        return [
            {"role": "system", "content": self.build_system_prompt(session_key, creative=creative)},
            *history,
            {"role": "user", "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        if not media:
            return text
        images = []
        attachments = []
        for path in media:
            file_path = Path(path)
            if not file_path.is_file():
                continue
            raw = file_path.read_bytes()
            mime = detect_image_mime(raw)
            if mime:
                import base64

                images.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"
                        },
                    }
                )
            else:
                attachments.append(str(file_path))
        parts: list[dict[str, Any]] = []
        if attachments:
            parts.append(
                {
                    "type": "text",
                    "text": "Attached files:\n" + "\n".join(f"- {item}" for item in attachments),
                }
            )
        parts.extend(images)
        parts.append({"type": "text", "text": text})
        return parts

    def add_tool_result(
        self, messages: list[dict[str, Any]], tool_call_id: str, tool_name: str, result: str
    ) -> list[dict[str, Any]]:
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        messages.append(
            build_assistant_message(
                content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
            )
        )
        return messages


class AgentLoop:
    _TOOL_RESULT_MAX_CHARS = 16_000
    _MODEL_ALIASES = {
        "minimax": "minimax-m2.7",
        "kimi": "kimi-k2.5",
        "glm": "glm-5",
    }

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        config: MikeConfig,
        store: ChatStore,
        research: ResearchManager,
        writing: Any | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.config = config
        self.store = store
        self.research = research
        self.model = config.default_model or DEFAULT_MODEL
        self.context = ContextBuilder(store)
        self.archiver = ArchiveManager(store, provider, self._get_effective_model)
        self.tools = ToolRegistry()
        self.writing = writing
        self._running = False
        self._processing_lock = asyncio.Lock()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._attached_mode: dict[str, bool] = {}
        self._context_offsets: dict[str, int] = {}
        self._save_queue: asyncio.Queue[tuple[str, str, str, int]] = asyncio.Queue()
        self._save_revisions: dict[str, int] = {}
        self._save_worker: asyncio.Task | None = None
        self._register_tools()

    def _register_tools(self) -> None:
        root = self.config.project_root_path
        allowed = root if self.config.restrict_shell_to_project else None
        self.tools.register(ReadFileTool(workspace=root, allowed_dir=allowed))
        self.tools.register(WriteFileTool(workspace=root, allowed_dir=allowed))
        self.tools.register(EditFileTool(workspace=root, allowed_dir=allowed))
        self.tools.register(ListDirTool(workspace=root, allowed_dir=allowed))
        self.tools.register(
            ExecTool(
                timeout=self.config.command_timeout,
                working_dir=str(root),
                restrict_to_workspace=self.config.restrict_shell_to_project,
            )
        )
        self.tools.register(
            WebSearchTool(
                cli_bin=self.config.opencode_server_bin,
                attach_url=self.config.opencode_server_url,
                provider_id=self.config.opencode_model_provider_id,
            )
        )
        self.tools.register(WebFetchTool(proxy=self.config.telegram_proxy))
        self.tools.register(ReadMemoryTool(self.store.memory_path))
        self.tools.register(SearchHistoryTool(self.store.history_index_path))
        self.tools.register(GetHistoryConversationTool(self.store.history_record_path))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(OpenCodeDelegateTool(manager=self.research))
        self.tools.register(ResearchTool(manager=self.research))

    async def run(self) -> None:
        self._running = True
        self._ensure_save_worker()
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
                continue
            if cmd == "/restart":
                await self._handle_restart(msg)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(
                lambda t, key=msg.session_key: (
                    self._active_tasks.get(key, []) and self._active_tasks[key].remove(t)
                    if t in self._active_tasks.get(key, [])
                    else None
                )
            )

    def stop(self) -> None:
        self._running = False
        if self._save_worker and not self._save_worker.done():
            self._save_worker.cancel()

    async def _handle_stop(self, msg: InboundMessage) -> None:
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for task in tasks if not task.done() and task.cancel())
        for task in tasks:
            try:
                await task
            except Exception:
                pass
        runtime_key = msg.session_key
        active_session_id = self.store.resolve_active_session(runtime_key)
        research_cancelled = await self.research.cancel_by_session(active_session_id)
        total = cancelled + research_cancelled
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Stopped {total} task(s)." if total else "No active task to stop.",
            )
        )

    async def _handle_restart(self, msg: InboundMessage) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")
        )

        async def do_restart() -> None:
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.create_task(do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    )
                )

    def _set_tool_context(
        self, channel: str, chat_id: str, message_id: str | None = None, model: str | None = None
    ) -> None:
        for name in ("message", "web_search", "opencode_delegate", "research"):
            tool = self.tools.get(name)
            setter = getattr(tool, "set_context", None)
            if not callable(setter):
                continue
            if name == "message":
                setter(channel, chat_id, message_id)
            else:
                setter(channel, chat_id, model)

    def _get_effective_model(self, session: ChatSession) -> str:
        if session.current_model and session.current_model in SUPPORTED_MODELS:
            return session.current_model
        return self.model

    def _has_vision_content(self, msg: InboundMessage) -> bool:
        for item in msg.media:
            lower = item.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                return True
        return False

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        def fmt(call: Any) -> str:
            args = call.arguments or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) and args else None
            if not isinstance(val, str):
                return call.name
            return f'{call.name}("{val[:40]}...")' if len(val) > 40 else f'{call.name}("{val}")'

        return ", ".join(fmt(call) for call in tool_calls)

    def _ensure_save_worker(self) -> None:
        if self._save_worker and not self._save_worker.done():
            return
        self._save_worker = asyncio.create_task(self._save_worker_loop())

    def _touch_revision(self, session_id: str) -> int:
        revision = self._save_revisions.get(session_id, 0) + 1
        self._save_revisions[session_id] = revision
        return revision

    async def _enqueue_background_save(self, session_id: str, channel: str, chat_id: str) -> None:
        self._ensure_save_worker()
        revision = self._touch_revision(session_id)
        await self._save_queue.put((session_id, channel, chat_id, revision))

    async def _save_worker_loop(self) -> None:
        while True:
            try:
                session_id, channel, chat_id, revision = await asyncio.wait_for(
                    self._save_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                if not self._running and self._save_queue.empty():
                    break
                continue
            try:
                latest = self._save_revisions.get(session_id, 0)
                if revision != latest:
                    continue
                session = self.store.load_session_record(session_id)
                if not session or not session.has_meaningful_content():
                    continue
                summary, memory_update = await self.archiver.summarize_session(session)
                if revision != self._save_revisions.get(session_id, 0):
                    continue
                session.summary = summary
                session.updated_at = datetime.now().isoformat()
                self.store.memory_path().write_text(memory_update, encoding="utf-8")
                self.store.save_session_record(session, channel=channel, chat_id=chat_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Background session save failed for {}", session_id)
            finally:
                self._save_queue.task_done()

    @staticmethod
    def _session_preview(summary: str) -> str:
        text = (summary or "").strip()
        if not text:
            return "(No summary yet)"
        line = text.splitlines()[0].strip()
        return line[:120] + ("..." if len(line) > 120 else "")

    @staticmethod
    def _relative_time(iso_text: str | None) -> str:
        if not iso_text:
            return "just now"
        try:
            dt = datetime.fromisoformat(iso_text)
        except Exception:
            return "just now"
        delta = datetime.now(dt.tzinfo) - dt
        seconds = max(0, int(delta.total_seconds()))
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"

    def _runtime_key(self, msg: InboundMessage, session_key: str | None = None) -> str:
        return (session_key or msg.session_key).strip()

    def _active_session(self, runtime_key: str) -> ChatSession:
        active_id = self.store.resolve_active_session(runtime_key)
        return self.store.get_or_create_session_record(active_id)

    @staticmethod
    def _has_meaningful_messages(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant", "system"} and content:
                return True
        return False

    def _session_offset(self, session_id: str, total_messages: int) -> int:
        offset = self._context_offsets.get(session_id, 0)
        if offset < 0 or offset > total_messages:
            return 0
        return offset

    @staticmethod
    def _is_positive_int(text: str) -> bool:
        return text.isdigit() and int(text) > 0

    def _format_sessions_page(self, entries: list[dict[str, Any]], page: int) -> str:
        if not entries:
            return "No saved sessions yet."
        per_page = 5
        total_pages = (len(entries) + per_page - 1) // per_page
        if page > total_pages:
            return f"Only {total_pages} pages available."
        start = (page - 1) * per_page
        selected = entries[start : start + per_page]
        lines: list[str] = []
        for item in selected:
            sid = str(item.get("id") or item.get("session_id") or "")
            raw_meta = item.get("metadata")
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            updated = str((meta or {}).get("updated_at") or item.get("archived_at") or "")
            summary = str(item.get("summary") or "")
            lines.append(f"{sid} · {self._relative_time(updated)}")
            lines.append(f"  {self._session_preview(summary)}")
            lines.append("")
        footer = f"Page {page} of {total_pages}"
        if page < total_pages:
            footer += f" · /sessions {page + 1} for older"
        lines.append(footer)
        return "\n".join(lines).strip()

    def _build_temp_prompt(self) -> str:
        soul_path = self.store.soul_path()
        soul = soul_path.read_text(encoding="utf-8").strip() if soul_path.exists() else ""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        parts = [
            "# Mike",
            "You are Mike, a focused personal assistant bot.",
            f"Current local time: {now}",
        ]
        if soul:
            parts.extend(["", "## SOUL", soul])
        return "\n".join(parts)

    async def _run_temp_message(
        self, query: str, msg: InboundMessage, model: str | None = None
    ) -> OutboundMessage:
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"), model)
        messages = [
            {"role": "system", "content": self._build_temp_prompt()},
            {
                "role": "user",
                "content": f"{ContextBuilder._build_runtime_context(msg.channel, msg.chat_id)}\n\n{query}",
            },
        ]
        final_content, _, _ = await self._run_agent_loop(messages, model=model)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content or "I've completed processing but have no response to give.",
            metadata=msg.metadata or {},
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict[str, Any]],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        model: str | None = None,
    ) -> tuple[str | None, list[str], list[dict[str, Any]]]:
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        effective_model = model or self.model or DEFAULT_MODEL
        reasoning = build_reasoning_kwargs(effective_model)
        max_tokens = clamp_max_tokens(effective_model, self.config.max_tokens)
        while iteration < self.config.max_tool_iterations:
            iteration += 1
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=effective_model,
                max_tokens=max_tokens,
                thinking=reasoning.get("thinking"),
                reasoning_effort=reasoning.get("reasoning_effort"),
            )
            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought and self.config.send_progress:
                        await on_progress(thought)
                    if self.config.send_tool_hints:
                        await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)
                tool_call_dicts = [call.to_openai_tool_call() for call in response.tool_calls]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                continue
            clean = self._strip_think(response.content)
            if response.finish_reason == "error":
                final_content = clean or "Sorry, I encountered an error calling the AI model."
                break
            messages = self.context.add_assistant_message(
                messages,
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            final_content = clean
            break
        if final_content is None and iteration >= self.config.max_tool_iterations:
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.config.max_tool_iterations}) "
                "without completing the task."
            )
        return final_content, tools_used, messages

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        if msg.channel == "system":
            if (msg.metadata or {}).get("_task_result"):
                session_id = msg.chat_id.strip().lower()
                session = self.store.get_or_create_session_record(session_id)
                session.add_message(
                    "system",
                    msg.content,
                    task_id=(msg.metadata or {}).get("task_id"),
                    task_label=(msg.metadata or {}).get("task_label"),
                    task_status=(msg.metadata or {}).get("task_status"),
                )
                self.store.save_session_record(session, channel="system", chat_id=msg.chat_id)
                return None
        runtime_key = self._runtime_key(msg, session_key)
        session = self._active_session(runtime_key)
        active_session_id = session.key
        raw_content = msg.content.strip()
        cmd = raw_content.lower()

        if cmd == "/new":
            offset = self._session_offset(active_session_id, len(session.messages))
            active_slice = session.messages[offset:]
            if not self._has_meaningful_messages(active_slice):
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Fresh session is already active.",
                )
            self.store.save_session_record(session, channel=msg.channel, chat_id=msg.chat_id)
            await self._enqueue_background_save(active_session_id, msg.channel, msg.chat_id)
            if self._attached_mode.get(runtime_key):
                self._context_offsets[active_session_id] = len(session.messages)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Session `{active_session_id}` saved. Fresh context started on the same session.",
                )
            new_session_id = self.store.clear_active_session(runtime_key)
            self._attached_mode[runtime_key] = False
            self.store.get_or_create_session_record(new_session_id)
            self._context_offsets[new_session_id] = 0
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Session `{active_session_id}` saved. New session `{new_session_id}` started.",
            )

        if cmd == "/clear":
            session = self.store.reset_session_record(active_session_id, preserve_model=True)
            self._context_offsets[active_session_id] = 0
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="Chat cleared."
            )

        if cmd.startswith("/attach"):
            parts = raw_content.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /attach <id>",
                )
            target = parts[1].strip().lower()
            if target == active_session_id:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"You're already in session `{target}`.",
                )
            if not self.store.session_record_exists(target):
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Session `{target}` not found.",
                )
            self.store.set_active_session(runtime_key, target)
            self._attached_mode[runtime_key] = True
            loaded = self.store.get_or_create_session_record(target)
            self._context_offsets[target] = 0
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Attached session `{target}`.",
            )

        if cmd.startswith("/fork"):
            parts = raw_content.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /fork <id>",
                )
            target = parts[1].strip().lower()
            source = self.store.load_session_record(target)
            if not source:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Session `{target}` not found.",
                )
            new_id = self.store.new_session_id()
            now = datetime.now().isoformat()
            forked = ChatSession(
                key=new_id,
                summary=source.summary,
                current_model=source.current_model,
                messages=json.loads(json.dumps(source.messages, ensure_ascii=False)),
                created_at=now,
                updated_at=now,
            )
            self.store.save_session_record(forked, channel=msg.channel, chat_id=msg.chat_id)
            self.store.set_active_session(runtime_key, new_id)
            self._attached_mode[runtime_key] = False
            self._context_offsets[new_id] = 0
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Forked session `{target}` as `{new_id}`.",
            )

        if cmd == "/sessions" or cmd.startswith("/sessions "):
            arg = raw_content[len("/sessions") :].strip()
            if not arg:
                entries = self.store.list_session_entries()
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=self._format_sessions_page(entries, 1),
                )
            if arg.lower().startswith("search"):
                query = arg[6:].strip()
                if not query:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Usage: /sessions search <query>",
                    )
                matches = self.store.search_session_entries(query, limit=5)
                if not matches:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"No sessions matching `{query}`.",
                    )
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=self._format_sessions_page(matches, 1),
                )
            if not self._is_positive_int(arg):
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /sessions [page]",
                )
            page = int(arg)
            entries = self.store.list_session_entries()
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._format_sessions_page(entries, page),
            )

        if cmd.startswith("/temp") or cmd.startswith("/btw"):
            prefix = "/temp" if cmd.startswith("/temp") else "/btw"
            query = raw_content[len(prefix) :].strip()
            if not query:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Usage: {prefix} <question>",
                )
            return await self._run_temp_message(
                query,
                msg,
                model=self._get_effective_model(session),
            )

        if cmd == "/help":
            lines = [
                "Mike commands:",
                "/new - Start a new conversation",
                "/clear - Clear chat instantly",
                "/attach <id> - Resume a saved session",
                "/fork <id> - Fork a saved session",
                "/sessions [page] - List saved sessions",
                "/sessions search <query> - Search sessions",
                "/temp <question> - One-off question with SOUL only",
                "/btw <question> - Alias for /temp",
                "/stop - Stop the current task",
                "/restart - Restart the bot",
                "/help - Show available commands",
                "/model - Show current model and available options",
                "/model <name> - Switch to a different model",
                "/model reset - Reset to default model",
                "/research <task> - Run a background research task",
                "/status - Show running background tasks",
                "/context <text> - Add context to a running task",
                "/write <directive> - Write a creative piece now",
                "/story list - List active story projects",
                "/story start <directive> - Start a chaptered story",
                "/story next <story_id> - Write next chapter",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines)
            )

        if cmd == "/model" or cmd.startswith("/model "):
            return await self._handle_model_command(msg, session)

        if cmd.startswith("/research"):
            task = msg.content[len("/research") :].strip()
            if not task:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /research <task description>",
                )
            result = await self.research.start_task(
                query=task,
                session_key=active_session_id,
                channel=msg.channel,
                chat_id=msg.chat_id,
                model=self._get_effective_model(session),
                kind="research",
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

        if cmd == "/status":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self.research.format_status(active_session_id),
            )

        if cmd.startswith("/context"):
            extra = msg.content[len("/context") :].strip()
            if not extra:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /context <extra information>",
                )
            result = await self.research.inject_context(active_session_id, extra)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

        if cmd.startswith("/write"):
            if not self.writing:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Writing mode is not available.",
                )
            directive = msg.content[len("/write") :].strip()
            result = await self.writing.write_on_demand(
                directive, active_session_id, msg.channel, msg.chat_id
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

        if cmd.startswith("/story"):
            if not self.writing:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Story mode is not available.",
                )
            parts = msg.content.strip().split(maxsplit=2)
            if len(parts) == 1 or parts[1].lower() == "list":
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=self.writing.format_story_list(),
                )
            action = parts[1].lower()
            if action == "start":
                directive = parts[2] if len(parts) > 2 else ""
                result = await self.writing.start_story(
                    directive, active_session_id, msg.channel, msg.chat_id
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)
            if action == "next":
                story_id = parts[2].strip() if len(parts) > 2 else ""
                if not story_id:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Usage: /story next <story_id>",
                    )
                result = await self.writing.continue_story(
                    story_id, active_session_id, msg.channel, msg.chat_id
                )
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Usage: /story list | /story start <directive> | /story next <story_id>",
            )

        creative = bool((msg.metadata or {}).get("_creative_soul"))

        model_override = (msg.metadata or {}).get("_model_override")
        effective_model = (
            model_override
            if isinstance(model_override, str) and model_override in SUPPORTED_MODELS
            else self._get_effective_model(session)
        )
        if self._has_vision_content(msg) and not model_supports_vision(effective_model):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Model {effective_model} does not support images. Switch to {DEFAULT_MODEL} with /model {DEFAULT_MODEL}.",
            )

        self._set_tool_context(
            msg.channel, msg.chat_id, msg.metadata.get("message_id"), effective_model
        )
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.start_turn()
        history = session.history()
        offset = self._session_offset(active_session_id, len(history))
        history = history[offset:]
        initial_messages = self.context.build_messages(
            active_session_id,
            history,
            msg.content,
            media=msg.media or None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            creative=creative,
        )

        async def bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta
                )
            )

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or bus_progress,
            model=effective_model,
        )
        final_content = final_content or "I've completed processing but have no response to give."
        self._save_turn(session, all_msgs, 1 + len(history))
        self.store.save_session_record(session, channel=msg.channel, chat_id=msg.chat_id)
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return None
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
        )

    async def _handle_model_command(
        self, msg: InboundMessage, session: ChatSession
    ) -> OutboundMessage:
        parts = msg.content.strip().split(maxsplit=1)
        if len(parts) == 1:
            current = self._get_effective_model(session)
            lines = [f"Current model: {current}", "", "Available models:"]
            for name, info in SUPPORTED_MODELS.items():
                vision_badge = " [vision]" if info["vision"] else ""
                marker = "-> " if name == current else "   "
                lines.append(f"{marker}{name}{vision_badge} - {info['description']}")
            lines.append("")
            lines.append("Usage: /model <name> to switch, /model reset to restore default")
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines)
            )
        subcmd = parts[1].lower().strip()
        subcmd = self._MODEL_ALIASES.get(subcmd, subcmd)
        if subcmd == "reset":
            old = self._get_effective_model(session)
            session.current_model = None
            session.updated_at = datetime.now().isoformat()
            self.store.save_session_record(session, channel=msg.channel, chat_id=msg.chat_id)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Model reset from {old} to default ({self._get_effective_model(session)}).",
            )
        if subcmd in SUPPORTED_MODELS:
            old = self._get_effective_model(session)
            session.current_model = subcmd
            session.updated_at = datetime.now().isoformat()
            self.store.save_session_record(session, channel=msg.channel, chat_id=msg.chat_id)
            note = " Supports images." if SUPPORTED_MODELS[subcmd]["vision"] else " Text-only."
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Model switched from {old} to {subcmd}.{note}",
            )
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"Unknown model: '{subcmd}'. Available: {', '.join(SUPPORTED_MODELS)}",
        )

    def _save_turn(self, session: ChatSession, messages: list[dict[str, Any]], skip: int) -> None:
        for message in messages[skip:]:
            entry = dict(message)
            role = entry.get("role")
            content = entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
            if (
                role == "tool"
                and isinstance(content, str)
                and len(content) > self._TOOL_RESULT_MAX_CHARS
            ):
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            if role == "user":
                if isinstance(content, str) and content.startswith(
                    ContextBuilder._RUNTIME_CONTEXT_TAG
                ):
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for part in content:
                        if (
                            part.get("type") == "text"
                            and isinstance(part.get("text"), str)
                            and part["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
                        ):
                            continue
                        if part.get("type") == "image_url" and part.get("image_url", {}).get(
                            "url", ""
                        ).startswith("data:image/"):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(part)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now().isoformat()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        creative: bool = False,
        model: str | None = None,
    ) -> str:
        metadata: dict[str, Any] = {}
        if creative:
            metadata["_creative_soul"] = True
        if model:
            metadata["_model_override"] = model
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            session_key_override=session_key,
            metadata=metadata,
        )
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress
        )
        return response.content if response else ""

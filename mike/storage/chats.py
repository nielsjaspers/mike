"""Chat state persistence for Mike."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mike.bootstrap import ensure_chat_files, seed_research_skill
from mike.common import ensure_dir, safe_filename, timestamp
from mike.config import MikeConfig
from mike.storage.files import chat_root


@dataclass
class ChatSession:
    key: str
    current_model: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=timestamp)
    updated_at: str = field(default_factory=timestamp)

    def add_message(self, role: str, content: Any, **extra: Any) -> None:
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **extra}
        )
        self.updated_at = timestamp()

    def history(self, limit: int = 500) -> list[dict[str, Any]]:
        items = self.messages[-limit:]
        result: list[dict[str, Any]] = []
        for item in items:
            clean = {"role": item["role"], "content": item.get("content")}
            for key in (
                "tool_calls",
                "tool_call_id",
                "name",
                "reasoning_content",
                "thinking_blocks",
            ):
                if key in item:
                    clean[key] = item[key]
            result.append(clean)
        return result

    def clear(self) -> None:
        self.messages = []
        self.updated_at = timestamp()


class ChatStore:
    def __init__(self, config: MikeConfig):
        self.config = config
        self.data_dir = ensure_dir(config.data_dir_path)
        self._cache: dict[str, ChatSession] = {}

    def root(self, session_key: str) -> Path:
        root = chat_root(self.data_dir, session_key)
        ensure_chat_files(root)
        seed_research_skill(self.config, root)
        return root

    def state_path(self, session_key: str) -> Path:
        return self.root(session_key) / "state.json"

    def uploads_dir(self, session_key: str) -> Path:
        return ensure_dir(self.root(session_key) / "uploads")

    def file_path(self, session_key: str, name: str) -> Path:
        return self.root(session_key) / name

    def get(self, session_key: str) -> ChatSession:
        if session_key in self._cache:
            return self._cache[session_key]
        path = self.state_path(session_key)
        if not path.exists():
            session = ChatSession(key=session_key)
            self.save(session)
            return session
        data = json.loads(path.read_text(encoding="utf-8"))
        session = ChatSession(**data)
        self._cache[session_key] = session
        return session

    def save(self, session: ChatSession) -> None:
        path = self.state_path(session.key)
        path.write_text(
            json.dumps(
                {
                    "key": session.key,
                    "current_model": session.current_model,
                    "messages": session.messages,
                    "created_at": session.created_at,
                    "updated_at": timestamp(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._cache[session.key] = session

    def append_history(self, session_key: str, text: str) -> None:
        path = self.file_path(session_key, "HISTORY.md")
        stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {text.strip()}\n")

    def save_upload(self, session_key: str, filename: str, data: bytes) -> str:
        safe = safe_filename(filename or "upload.bin")
        path = self.uploads_dir(session_key) / safe
        path.write_bytes(data)
        return str(path)

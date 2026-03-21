"""Chat state persistence for Mike."""

from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mike.bootstrap import ensure_root, ensure_session_dirs
from mike.common import ensure_dir, safe_filename, timestamp
from mike.config import MikeConfig
from mike.storage.files import history_records_root, history_root, session_root


SESSION_ID_PATTERN = re.compile(r"^[a-z0-9]{5}$")


@dataclass
class ChatSession:
    key: str
    summary: str = ""
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
        now = timestamp()
        self.created_at = now
        self.updated_at = now

    def has_meaningful_content(self) -> bool:
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant", "system"} and content:
                return True
        return False


class ChatStore:
    def __init__(self, config: MikeConfig):
        self.config = config
        self.data_dir = ensure_root(config)
        self._cache: dict[str, ChatSession] = {}
        self._active_session_by_runtime: dict[str, str] = {}

    @property
    def shared_root(self) -> Path:
        return self.data_dir

    def session_root(self, session_key: str) -> Path:
        root = session_root(self.data_dir, session_key)
        ensure_session_dirs(root)
        return root

    def state_path(self, session_key: str) -> Path:
        return self.session_root(session_key) / "active.json"

    def uploads_dir(self, session_key: str) -> Path:
        return ensure_dir(self.session_root(session_key) / "uploads")

    def shared_file(self, name: str) -> Path:
        return self.shared_root / name

    def memory_path(self) -> Path:
        return self.shared_file("MEMORY.md")

    def soul_path(self) -> Path:
        return self.shared_file("SOUL.md")

    def user_path(self) -> Path:
        return self.shared_file("USER.md")

    def skills_root(self) -> Path:
        return self.shared_root / "skills"

    def history_index_path(self) -> Path:
        return history_root(self.data_dir) / "index.json"

    def history_records_root(self) -> Path:
        return history_records_root(self.data_dir)

    def history_record_path(self, archive_id: str) -> Path:
        return self.history_records_root() / f"{safe_filename(archive_id)}.json"

    def get(self, session_key: str) -> ChatSession:
        if session_key in self._cache:
            return self._cache[session_key]
        path = self.state_path(session_key)
        if not path.exists():
            session = ChatSession(key=session_key)
            self.save(session)
            return session
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" not in data:
            data["summary"] = ""
        session = ChatSession(**data)
        self._cache[session_key] = session
        return session

    def save(self, session: ChatSession) -> None:
        path = self.state_path(session.key)
        payload = {
            "key": session.key,
            "summary": session.summary,
            "current_model": session.current_model,
            "messages": session.messages,
            "created_at": session.created_at,
            "updated_at": timestamp(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._cache[session.key] = session

    def reset(self, session_key: str, preserve_model: bool = True) -> ChatSession:
        current = (
            self.get(session_key)
            if session_key in self._cache or self.state_path(session_key).exists()
            else None
        )
        session = ChatSession(
            key=session_key,
            current_model=current.current_model if preserve_model and current else None,
        )
        self.save(session)
        self._cache[session_key] = session
        return session

    def save_upload(self, session_key: str, filename: str, data: bytes) -> str:
        safe = safe_filename(filename or "upload.bin")
        path = self.uploads_dir(session_key) / safe
        path.write_bytes(data)
        return str(path)

    @staticmethod
    def looks_like_session_id(value: str) -> bool:
        return bool(SESSION_ID_PATTERN.match(value.strip().lower()))

    def _session_record_path(self, session_id: str) -> Path:
        return self.history_record_path(session_id)

    def _normalize_session_id(self, session_id: str) -> str:
        return session_id.strip().lower()

    def new_session_id(self) -> str:
        alphabet = string.ascii_lowercase + string.digits
        for _ in range(500):
            candidate = "".join(random.choice(alphabet) for _ in range(5))
            if not self._session_record_path(candidate).exists() and candidate not in self._cache:
                return candidate
        raise RuntimeError("Failed to generate unique session id")

    def resolve_active_session(self, runtime_key: str) -> str:
        key = runtime_key.strip()
        if self.looks_like_session_id(key):
            return self._normalize_session_id(key)
        if key in self._active_session_by_runtime:
            return self._active_session_by_runtime[key]
        session_id = self.new_session_id()
        self._active_session_by_runtime[key] = session_id
        return session_id

    def set_active_session(self, runtime_key: str, session_id: str) -> str:
        sid = self._normalize_session_id(session_id)
        self._active_session_by_runtime[runtime_key.strip()] = sid
        return sid

    def clear_active_session(self, runtime_key: str) -> str:
        key = runtime_key.strip()
        session_id = self.new_session_id()
        self._active_session_by_runtime[key] = session_id
        return session_id

    def current_active_session(self, runtime_key: str) -> str | None:
        return self._active_session_by_runtime.get(runtime_key.strip())

    def resolve_runtime_session_id(self, runtime_key: str) -> str | None:
        key = runtime_key.strip()
        if self.looks_like_session_id(key):
            return self._normalize_session_id(key)
        return self._active_session_by_runtime.get(key)

    def load_runtime_session(self, runtime_key: str) -> ChatSession | None:
        session_id = self.resolve_runtime_session_id(runtime_key)
        if not session_id:
            return None
        return self.load_session_record(session_id)

    @staticmethod
    def _record_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
        msgs = record.get("messages")
        if isinstance(msgs, list):
            return msgs
        legacy = record.get("full_chat_log")
        if isinstance(legacy, list):
            return legacy
        return []

    @staticmethod
    def _record_summary(record: dict[str, Any]) -> str:
        summary = record.get("summary")
        return str(summary).strip() if isinstance(summary, str) else ""

    def load_session_record(self, session_id: str) -> ChatSession | None:
        sid = self._normalize_session_id(session_id)
        if sid in self._cache:
            return self._cache[sid]
        path = self._session_record_path(sid)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        created_at = str(
            metadata.get("created_at")
            or metadata.get("started_at")
            or data.get("created_at")
            or timestamp()
        )
        updated_at = str(
            metadata.get("updated_at")
            or metadata.get("ended_at")
            or data.get("updated_at")
            or timestamp()
        )
        session = ChatSession(
            key=sid,
            summary=self._record_summary(data),
            current_model=str(data.get("current_model") or "").strip() or None,
            messages=self._record_messages(data),
            created_at=created_at,
            updated_at=updated_at,
        )
        self._cache[sid] = session
        return session

    def get_or_create_session_record(self, session_id: str) -> ChatSession:
        sid = self._normalize_session_id(session_id)
        existing = self.load_session_record(sid)
        if existing is not None:
            return existing
        session = ChatSession(key=sid)
        self._cache[sid] = session
        return session

    def reset_session_record(self, session_id: str, preserve_model: bool = True) -> ChatSession:
        sid = self._normalize_session_id(session_id)
        current = self.load_session_record(sid)
        model = current.current_model if preserve_model and current else None
        self.delete_session_record(sid)
        session = ChatSession(key=sid, current_model=model)
        self._cache[sid] = session
        return session

    def session_record_exists(self, session_id: str) -> bool:
        sid = self._normalize_session_id(session_id)
        return self._session_record_path(sid).exists()

    def _upsert_history_index_entry(self, entry: dict[str, Any]) -> None:
        index = []
        path = self.history_index_path()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    index = raw
            except Exception:
                index = []
        target_id = str(entry.get("id", "")).strip()
        replaced = False
        for idx, item in enumerate(index):
            if str(item.get("id", "")).strip() == target_id:
                index[idx] = entry
                replaced = True
                break
        if not replaced:
            index.append(entry)
        path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete_session_record(self, session_id: str) -> None:
        sid = self._normalize_session_id(session_id)
        path = self._session_record_path(sid)
        if path.exists():
            path.unlink()
        self._cache.pop(sid, None)
        index_path = self.history_index_path()
        if not index_path.exists():
            return
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, list):
            return
        filtered = [
            row
            for row in raw
            if not (isinstance(row, dict) and str(row.get("id", "")).strip() == sid)
        ]
        if len(filtered) != len(raw):
            index_path.write_text(
                json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def save_session_record(self, session: ChatSession, *, channel: str, chat_id: str) -> None:
        sid = self._normalize_session_id(session.key)
        session.key = sid
        message_count = len(session.messages)
        models_used = sorted(
            {
                str(message.get("model", "")).strip()
                for message in session.messages
                if isinstance(message, dict) and str(message.get("model", "")).strip()
            }
        )
        if session.current_model and session.current_model not in models_used:
            models_used.append(session.current_model)
        metadata = {
            "kind": "session",
            "channel": channel,
            "chat_id": chat_id,
            "session_key": sid,
            "message_count": message_count,
            "models_used": models_used,
            "started_at": session.created_at,
            "ended_at": session.updated_at,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }
        summary = (session.summary or "").strip()
        record = {
            "id": sid,
            "session_id": sid,
            "title": f"Session {sid}",
            "summary": summary,
            "messages": session.messages,
            "full_chat_log": session.messages,
            "current_model": session.current_model,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": message_count,
            "metadata": metadata,
        }
        self._session_record_path(sid).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._upsert_history_index_entry(
            {
                "id": sid,
                "kind": "session",
                "title": record["title"],
                "summary": summary,
                "archived_at": session.updated_at,
                "metadata": {
                    "channel": channel,
                    "chat_id": chat_id,
                    "message_count": message_count,
                    "models_used": models_used,
                    "started_at": session.created_at,
                    "ended_at": session.updated_at,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                },
            }
        )

    def _session_index_entries(self) -> list[dict[str, Any]]:
        path = self.history_index_path()
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        result = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip().lower()
            if not item_id:
                continue
            kind = str(item.get("kind") or (item.get("metadata") or {}).get("kind") or "")
            if kind and kind != "session":
                continue
            if not self._session_record_path(item_id).exists():
                continue
            result.append(item)
        return sorted(
            result,
            key=lambda row: str(
                (row.get("metadata") or {}).get("updated_at") or row.get("archived_at") or ""
            ),
            reverse=True,
        )

    def list_session_entries(self) -> list[dict[str, Any]]:
        return self._session_index_entries()

    def search_session_entries(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        q = query.strip().lower()
        if not q:
            return []
        items = []
        for item in self._session_index_entries():
            summary = str(item.get("summary") or "")
            if q in summary.lower():
                items.append(item)
            if len(items) >= limit:
                break
        return items

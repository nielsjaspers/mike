"""Path helpers for Mike storage."""

from __future__ import annotations

from pathlib import Path

from mike.common import ensure_dir, safe_filename


def chat_root(data_dir: Path, session_key: str) -> Path:
    return ensure_dir(data_dir / "chats" / safe_filename(session_key.replace(":", "_")))


def task_root(data_dir: Path, task_id: str) -> Path:
    return ensure_dir(data_dir / "tasks" / safe_filename(task_id))

"""Data models for iterative research tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from nanobot.utils.helpers import timestamp


@dataclass
class ResearchTask:
    """Persisted research task state."""

    task_id: str
    session_key: str
    origin_channel: str
    origin_chat_id: str
    query: str
    backend: str = "opencode"
    title: str = "research"
    status: str = "queued"
    phase: str = "RECEIVED"
    model: str | None = None
    progress_summary: str | None = None
    final_summary: str | None = None
    final_artifact: str | None = None
    user_injections: list[str] = field(default_factory=list)
    backend_session_id: str | None = None
    tokens_used: int = 0
    error: str | None = None
    created_at: str = field(default_factory=timestamp)
    updated_at: str = field(default_factory=timestamp)

    def touch(self) -> None:
        self.updated_at = timestamp()

    def to_dict(self) -> dict[str, Any]:
        self.touch()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchTask":
        return cls(**dict(data))

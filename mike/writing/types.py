"""Data structures for Nocturne writing mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

PIECE_TYPES = ["poetry", "philosophy", "tech_speculation", "fiction"]


def _now() -> str:
    return datetime.now().isoformat()


@dataclass
class WorkMetadata:
    work_id: str
    title: str
    piece_type: str
    timestamp: str
    summary: str
    file_path: str
    story_id: str | None = None
    chapter: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkMetadata":
        return cls(**data)


@dataclass
class StoryState:
    story_id: str
    title: str
    directive: str
    genre: str
    premise: str
    characters: list[dict[str, Any]] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    outline: list[dict[str, Any]] = field(default_factory=list)
    chapter_summaries: list[str] = field(default_factory=list)
    chapters_completed: int = 0
    status: str = "active"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = _now()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryState":
        return cls(**data)


@dataclass
class WritingMetadata:
    last_piece_type_index: int = -1
    total_pieces: int = 0
    active_stories: list[str] = field(default_factory=list)
    works: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WritingMetadata":
        payload = dict(data)
        payload.setdefault("works", [])
        return cls(**payload)

    def next_piece_type(self) -> str:
        if not PIECE_TYPES:
            return "fiction"
        self.last_piece_type_index = (self.last_piece_type_index + 1) % len(PIECE_TYPES)
        return PIECE_TYPES[self.last_piece_type_index]

    def add_work(self, work: WorkMetadata, max_entries: int = 500) -> None:
        self.works.insert(0, work.to_dict())
        self.total_pieces += 1
        if len(self.works) > max_entries:
            self.works = self.works[:max_entries]

"""Persistence layer for Nocturne writing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from mike.common import ensure_dir
from mike.writing.types import StoryState, WorkMetadata, WritingMetadata


class WritingStore:
    def __init__(self, writing_root: Path):
        self.root = ensure_dir(writing_root)
        self.works_root = ensure_dir(self.root / "works")
        self.stories_root = ensure_dir(self.root / "stories")
        self.metadata_path = self.root / "metadata.json"

    def ensure_dirs(self) -> None:
        ensure_dir(self.root)
        ensure_dir(self.works_root)
        ensure_dir(self.stories_root)

    def load_metadata(self) -> WritingMetadata:
        if not self.metadata_path.exists():
            return WritingMetadata()
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return WritingMetadata()
        return WritingMetadata.from_dict(data)

    def save_metadata(self, metadata: WritingMetadata) -> None:
        self.metadata_path.write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_work(self, work: WorkMetadata, content: str) -> Path:
        path = self.root / work.file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def list_works(self, limit: int = 20) -> list[WorkMetadata]:
        metadata = self.load_metadata()
        result: list[WorkMetadata] = []
        for item in metadata.works[:limit]:
            try:
                result.append(WorkMetadata.from_dict(item))
            except Exception:
                continue
        return result

    def story_root(self, story_id: str) -> Path:
        return ensure_dir(self.stories_root / story_id)

    def story_state_path(self, story_id: str) -> Path:
        return self.story_root(story_id) / "state.json"

    def story_chapter_path(self, story_id: str, chapter: int) -> Path:
        return self.story_root(story_id) / f"chapter-{chapter:02d}.md"

    def load_story(self, story_id: str) -> StoryState | None:
        path = self.story_state_path(story_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return StoryState.from_dict(data)
        except Exception:
            return None

    def save_story(self, story: StoryState) -> None:
        path = self.story_state_path(story.story_id)
        path.write_text(json.dumps(story.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def save_story_chapter(self, story_id: str, chapter: int, content: str) -> Path:
        path = self.story_chapter_path(story_id, chapter)
        path.write_text(content, encoding="utf-8")
        return path

    def read_story_chapter(self, story_id: str, chapter: int) -> str:
        path = self.story_chapter_path(story_id, chapter)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def list_stories(self) -> list[StoryState]:
        result: list[StoryState] = []
        for state_path in self.stories_root.glob("*/state.json"):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                result.append(StoryState.from_dict(data))
            except Exception:
                continue
        return sorted(result, key=lambda item: item.updated_at, reverse=True)

"""Persistence for iterative research tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanobot.research.models import ResearchTask
from nanobot.utils.helpers import ensure_dir, timestamp


class ResearchPersistence:
    """Stores research task snapshots and event logs."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.root = ensure_dir(workspace / "research" / "tasks")

    def task_dir(self, task_id: str) -> Path:
        return ensure_dir(self.root / task_id)

    def snapshot_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "task.json"

    def events_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "events.jsonl"

    def artifacts_dir(self, task_id: str) -> Path:
        return ensure_dir(self.task_dir(task_id) / "artifacts")

    def injections_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "injections.jsonl"

    def save_task(self, task: ResearchTask) -> None:
        path = self.snapshot_path(task.task_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def append_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        entry = {
            "timestamp": timestamp(),
            "type": event_type,
            "payload": payload,
        }
        with self.events_path(task_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def append_injection(self, task_id: str, text: str) -> None:
        with self.injections_path(task_id).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"timestamp": timestamp(), "text": text}, ensure_ascii=False) + "\n"
            )

    def load_task(self, task_id: str) -> ResearchTask | None:
        path = self.snapshot_path(task_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ResearchTask.from_dict(data)

    def list_tasks(self) -> list[ResearchTask]:
        items: list[ResearchTask] = []
        for path in self.root.glob("*/task.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                items.append(ResearchTask.from_dict(data))
            except Exception:
                continue
        return sorted(items, key=lambda item: item.updated_at, reverse=True)

    def load_unfinished(self) -> list[ResearchTask]:
        return [
            item
            for item in self.list_tasks()
            if item.status
            in {"queued", "running", "planning", "researching", "evaluating", "synthesizing"}
        ]

    def write_artifact(self, task_id: str, name: str, content: str) -> str:
        path = self.artifacts_dir(task_id) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

"""JSON persistence for schedules and run logs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from mike.common import ensure_dir, timestamp
from mike.scheduling.types import RunRecord, RunStatus, ScheduleItem


class ScheduleStore:
    def __init__(self, data_dir: Path):
        self.data_dir = ensure_dir(data_dir)
        self._items_path = self.data_dir / "items.json"
        self._runs_path = self.data_dir / "runs.jsonl"

    def _load_items(self) -> list[dict]:
        if not self._items_path.exists():
            return []
        try:
            return json.loads(self._items_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_items(self, items: list[dict]) -> None:
        self._items_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def list(self) -> list[ScheduleItem]:
        items = self._load_items()
        return [ScheduleItem.from_dict(d) for d in items if d.get("deleted_at") is None]

    def list_all(self) -> list[ScheduleItem]:
        items = self._load_items()
        return [ScheduleItem.from_dict(d) for d in items]

    def get(self, schedule_id: str) -> ScheduleItem | None:
        for d in self._load_items():
            if d.get("id") == schedule_id:
                return ScheduleItem.from_dict(d)
        return None

    def save(self, item: ScheduleItem) -> None:
        items = self._load_items()
        updated = False
        for i, d in enumerate(items):
            if d.get("id") == item.id:
                items[i] = item.to_dict()
                updated = True
                break
        if not updated:
            items.append(item.to_dict())
        self._save_items(items)

    def delete(self, schedule_id: str) -> None:
        items = self._load_items()
        for d in items:
            if d.get("id") == schedule_id:
                d["deleted_at"] = timestamp()
        self._save_items(items)

    def new_id(self) -> str:
        return f"sc_{uuid.uuid4().hex[:8]}"

    def append_run(self, run: RunRecord) -> None:
        with self._runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")

    def list_runs(self, schedule_id: str | None = None, limit: int = 100) -> list[RunRecord]:
        if not self._runs_path.exists():
            return []
        records: list[RunRecord] = []
        with self._runs_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = RunRecord.from_dict(json.loads(line))
                    if schedule_id is None or record.schedule_id == schedule_id:
                        records.append(record)
                except Exception:
                    continue
        records.sort(key=lambda r: r.started_at, reverse=True)
        return records[:limit]

    def get_run(self, run_id: str) -> RunRecord | None:
        for record in self.list_runs(limit=1000):
            if record.run_id == run_id:
                return record
        return None

    def has_succeeded_run(self, schedule_id: str, occurrence_at_utc: str) -> bool:
        for record in self.list_runs(schedule_id, limit=1000):
            if record.schedule_id == schedule_id and record.occurrence_at_utc == occurrence_at_utc:
                if record.status == RunStatus.SUCCEEDED:
                    return True
        return False

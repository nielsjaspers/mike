"""Scheduling types for Mike."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def _ts() -> str:
    return datetime.utcnow().isoformat()


class ScheduleKind(str, Enum):
    REMINDER = "reminder"
    TASK = "task"


class ScheduleType(str, Enum):
    ONCE = "once"
    RECURRING = "recurring"


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRY_SCHEDULED = "retry_scheduled"
    SKIPPED_DUPLICATE = "skipped_duplicate"


@dataclass
class Delivery:
    channel: str = "telegram"
    chat_id: str = ""
    message_thread_id: int | None = None


@dataclass
class Execution:
    allow_delegate: bool = True
    explicit_outputs_only: bool = True
    model: str | None = None


@dataclass
class RecurrenceRule:
    kind: str = "daily"
    weekdays: list[int] = field(default_factory=list)
    time: str = "09:00"
    interval_hours: int | None = None
    interval_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecurrenceRule:
        return cls(**data)


@dataclass
class ScheduleItem:
    id: str
    kind: ScheduleKind
    title: str
    prompt: str
    enabled: bool = True
    timezone: str = "Europe/Amsterdam"
    schedule_type: ScheduleType = ScheduleType.ONCE
    run_at_utc: str | None = None
    recurrence_text: str | None = None
    recurrence_rule: RecurrenceRule | None = None
    next_run_at_utc: str | None = None
    last_run_at_utc: str | None = None
    delivery: Delivery = field(default_factory=Delivery)
    execution: Execution = field(default_factory=Execution)
    created_at: str = field(default_factory=_ts)
    updated_at: str = field(default_factory=_ts)
    deleted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["schedule_type"] = self.schedule_type.value
        data["delivery"] = self.delivery.__dict__
        data["execution"] = self.execution.__dict__
        if self.recurrence_rule is not None:
            data["recurrence_rule"] = self.recurrence_rule.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduleItem:
        data = dict(data)
        data["kind"] = ScheduleKind(data["kind"])
        data["schedule_type"] = ScheduleType(data["schedule_type"])
        data["delivery"] = Delivery(**data.get("delivery", {}))
        data["execution"] = Execution(**data.get("execution", {}))
        if data.get("recurrence_rule"):
            data["recurrence_rule"] = RecurrenceRule.from_dict(data["recurrence_rule"])
        return cls(**data)


@dataclass
class RunRecord:
    run_id: str
    schedule_id: str
    occurrence_at_utc: str
    attempt: int = 1
    status: RunStatus = RunStatus.RUNNING
    started_at: str = field(default_factory=_ts)
    finished_at: str | None = None
    error: str | None = None
    result_summary: str | None = None
    explicit_outputs: list[str] = field(default_factory=list)
    idempotency_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        data = dict(data)
        data["status"] = RunStatus(data["status"])
        return cls(**data)

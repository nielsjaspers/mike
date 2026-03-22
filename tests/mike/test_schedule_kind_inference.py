from __future__ import annotations

from mike.bus import MessageBus
from mike.config import MikeConfig
from mike.scheduling.manager import ScheduleManager
from mike.scheduling.store import ScheduleStore
from mike.scheduling.types import ScheduleKind


def make_manager(tmp_path):
    cfg = MikeConfig(
        data_dir=str(tmp_path / "mike-data"),
        project_root=str(tmp_path),
        schedule_enabled=True,
    )
    bus = MessageBus()
    store = ScheduleStore(tmp_path / "schedules")
    return ScheduleManager(cfg, bus, store), store


def test_tool_create_infers_reminder_from_text(tmp_path):
    manager, store = make_manager(tmp_path)

    result = manager.tool_create(
        text="please remind me in 2 minutes with the text tiny grad is cool",
        prompt="tiny grad is cool",
        when_text="in 2 minutes",
        recurrence_text=None,
        kind_str=None,
        channel="telegram",
        chat_id="123",
    )

    assert result["ok"] is True
    schedule_id = result["data"]["schedule_id"]
    item = store.get(schedule_id)
    assert item is not None
    assert item.kind == ScheduleKind.REMINDER


def test_tool_create_uses_explicit_task_kind(tmp_path):
    manager, store = make_manager(tmp_path)

    result = manager.tool_create(
        text="in 2 minutes run nightly sync",
        prompt="run nightly sync",
        when_text="in 2 minutes",
        recurrence_text=None,
        kind_str="task",
        channel="telegram",
        chat_id="123",
    )

    assert result["ok"] is True
    schedule_id = result["data"]["schedule_id"]
    item = store.get(schedule_id)
    assert item is not None
    assert item.kind == ScheduleKind.TASK

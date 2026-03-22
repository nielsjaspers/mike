from __future__ import annotations

from pathlib import Path

import pytest

from mike.bus import MessageBus
from mike.config import MikeConfig
from mike.scheduling.manager import ScheduleManager
from mike.scheduling.store import ScheduleStore
from mike.scheduling.types import (
    Delivery,
    Execution,
    RecurrenceRule,
    ScheduleItem,
    ScheduleKind,
    ScheduleType,
)


def make_manager(tmp_path):
    cfg = MikeConfig(
        data_dir=str(tmp_path / "mike-data"),
        project_root=str(tmp_path),
        schedule_enabled=True,
    )
    bus = MessageBus()
    store = ScheduleStore(tmp_path / "schedules")
    return ScheduleManager(cfg, bus, store), cfg, bus, store


class TestScheduleManager:
    def test_handle_command_list_empty(self, tmp_path):
        manager, _, _, _ = make_manager(tmp_path)
        result = manager.handle_command("list", "telegram", "123")
        assert "No schedules" in result

    def test_handle_command_help(self, tmp_path):
        manager, _, _, _ = make_manager(tmp_path)
        result = manager.handle_command("", "telegram", "123")
        assert "/schedule" in result

    def test_handle_command_add_relative(self, tmp_path):
        manager, _, _, store = make_manager(tmp_path)
        result = manager.handle_command("add in 5 minutes, remind me to stretch", "telegram", "123")
        assert "Created schedule" in result
        assert "sc_" in result

        schedule_id = result.split("[")[1].split("]")[0]
        item = store.get(schedule_id)
        assert item is not None
        assert "remind me to stretch" in item.prompt
        assert item.delivery.chat_id == "123"
        assert item.delivery.channel == "telegram"

    def test_handle_command_add_recurring(self, tmp_path):
        manager, _, _, store = make_manager(tmp_path)
        result = manager.handle_command("add every day at 09:00, daily standup", "telegram", "123")
        assert "Created schedule" in result
        schedule_id = result.split("[")[1].split("]")[0]
        item = store.get(schedule_id)
        assert item is not None
        assert item.schedule_type == ScheduleType.RECURRING
        assert item.recurrence_rule is not None
        assert item.recurrence_rule.kind == "daily"

    def test_handle_command_show_nonexistent(self, tmp_path):
        manager, _, _, _ = make_manager(tmp_path)
        result = manager.handle_command("show nonexistent", "telegram", "123")
        assert "not found" in result

    def test_handle_command_show(self, tmp_path):
        manager, _, _, store = make_manager(tmp_path)
        add_result = manager.handle_command("add in 10 minutes, test task", "telegram", "123")
        schedule_id = add_result.split("[")[1].split("]")[0]

        show_result = manager.handle_command(f"show {schedule_id}", "telegram", "123")
        assert schedule_id in show_result
        assert "test task" in show_result

    def test_handle_command_pause(self, tmp_path):
        manager, _, _, store = make_manager(tmp_path)
        add_result = manager.handle_command("add in 10 minutes, test task", "telegram", "123")
        schedule_id = add_result.split("[")[1].split("]")[0]

        pause_result = manager.handle_command(f"pause {schedule_id}", "telegram", "123")
        assert "Paused" in pause_result

        item = store.get(schedule_id)
        assert item.enabled is False

    def test_handle_command_resume(self, tmp_path):
        manager, _, _, store = make_manager(tmp_path)
        add_result = manager.handle_command("add in 10 minutes, test task", "telegram", "123")
        schedule_id = add_result.split("[")[1].split("]")[0]
        store.delete(schedule_id)

        item = store.get(schedule_id)
        item.enabled = False
        item.schedule_type = ScheduleType.RECURRING
        item.recurrence_rule = RecurrenceRule(kind="daily", time="09:00")
        item.next_run_at_utc = None
        store.save(item)

        resume_result = manager.handle_command(f"resume {schedule_id}", "telegram", "123")
        assert "Resumed" in resume_result

    def test_handle_command_delete(self, tmp_path):
        manager, _, _, store = make_manager(tmp_path)
        add_result = manager.handle_command("add in 10 minutes, test task", "telegram", "123")
        schedule_id = add_result.split("[")[1].split("]")[0]

        delete_result = manager.handle_command(f"delete {schedule_id}", "telegram", "123")
        assert "Deleted" in delete_result

        assert store.list() == []

    def test_handle_command_update_recurring(self, tmp_path):
        manager, _, _, store = make_manager(tmp_path)
        add_result = manager.handle_command("add in 10 minutes, original task", "telegram", "123")
        schedule_id = add_result.split("[")[1].split("]")[0]

        update_result = manager.handle_command(
            f"update {schedule_id} every weekday at 08:00, updated task",
            "telegram",
            "123",
        )
        assert "Updated" in update_result

        item = store.get(schedule_id)
        assert item.schedule_type == ScheduleType.RECURRING
        assert item.recurrence_rule.kind == "weekly"
        assert item.recurrence_rule.weekdays == [0, 1, 2, 3, 4]

    def test_handle_command_list_after_add(self, tmp_path):
        manager, _, _, _ = make_manager(tmp_path)
        manager.handle_command("add in 10 minutes, task one", "telegram", "123")
        manager.handle_command("add in 20 minutes, task two", "telegram", "123")

        list_result = manager.handle_command("list", "telegram", "123")
        assert "task one" in list_result
        assert "task two" in list_result

    def test_parse_output_block_with_json(self, tmp_path):
        manager, _, _, _ = make_manager(tmp_path)
        text = 'Done!\n\n```json\n{"summary": "Task completed.", "outputs": ["report.md"]}\n```'
        summary, outputs = manager._parse_output_block(text, [])
        assert "Task completed" in summary
        assert "report.md" in outputs

    def test_parse_output_block_no_json(self, tmp_path):
        manager, _, _, _ = make_manager(tmp_path)
        text = "Just a regular response without JSON."
        summary, outputs = manager._parse_output_block(text, [])
        assert summary == text
        assert outputs == []

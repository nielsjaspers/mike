from __future__ import annotations

from datetime import datetime

import pytest

from mike.scheduling.store import ScheduleStore
from mike.scheduling.types import (
    Delivery,
    Execution,
    RecurrenceRule,
    RunRecord,
    RunStatus,
    ScheduleItem,
    ScheduleKind,
    ScheduleType,
)


def make_store(tmp_path):
    return ScheduleStore(tmp_path / "schedules")


def make_item(
    schedule_id="sc_test123",
    kind=ScheduleKind.TASK,
    prompt="test prompt",
    schedule_type=ScheduleType.ONCE,
):
    return ScheduleItem(
        id=schedule_id,
        kind=kind,
        title="Test",
        prompt=prompt,
        schedule_type=schedule_type,
        delivery=Delivery(channel="telegram", chat_id="123"),
        execution=Execution(),
    )


class TestScheduleStore:
    def test_save_and_get(self, tmp_path):
        store = make_store(tmp_path)
        item = make_item()
        store.save(item)

        loaded = store.get("sc_test123")
        assert loaded is not None
        assert loaded.id == "sc_test123"
        assert loaded.prompt == "test prompt"

    def test_get_nonexistent(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get("nonexistent") is None

    def test_list_empty(self, tmp_path):
        store = make_store(tmp_path)
        assert store.list() == []

    def test_list_returns_only_active(self, tmp_path):
        store = make_store(tmp_path)
        item = make_item()
        store.save(item)
        store.delete("sc_test123")

        assert store.list() == []

    def test_list_all_includes_deleted(self, tmp_path):
        store = make_store(tmp_path)
        item = make_item()
        store.save(item)
        store.delete("sc_test123")

        all_items = store.list_all()
        assert len(all_items) == 1

    def test_delete_marks_deleted_at(self, tmp_path):
        store = make_store(tmp_path)
        item = make_item()
        store.save(item)
        store.delete("sc_test123")

        all_items = store.list_all()
        assert all_items[0].deleted_at is not None

    def test_new_id_format(self, tmp_path):
        store = make_store(tmp_path)
        id1 = store.new_id()
        id2 = store.new_id()
        assert id1.startswith("sc_")
        assert id2.startswith("sc_")
        assert id1 != id2

    def test_append_run_and_list(self, tmp_path):
        store = make_store(tmp_path)
        run = RunRecord(
            run_id="run_abc",
            schedule_id="sc_test123",
            occurrence_at_utc=datetime.utcnow().isoformat(),
        )
        store.append_run(run)

        runs = store.list_runs("sc_test123")
        assert len(runs) == 1
        assert runs[0].run_id == "run_abc"

    def test_has_succeeded_run(self, tmp_path):
        store = make_store(tmp_path)
        occurrence = datetime.utcnow().isoformat()

        run1 = RunRecord(
            run_id="run_1",
            schedule_id="sc_test",
            occurrence_at_utc=occurrence,
            status=RunStatus.SUCCEEDED,
        )
        store.append_run(run1)

        assert store.has_succeeded_run("sc_test", occurrence) is True
        assert store.has_succeeded_run("sc_test", "wrong_occurrence") is False

    def test_list_runs_filters_by_schedule(self, tmp_path):
        store = make_store(tmp_path)
        occurrence = datetime.utcnow().isoformat()

        run1 = RunRecord(
            run_id="run_1",
            schedule_id="sc_test",
            occurrence_at_utc=occurrence,
        )
        run2 = RunRecord(
            run_id="run_2",
            schedule_id="sc_other",
            occurrence_at_utc=occurrence,
        )
        store.append_run(run1)
        store.append_run(run2)

        runs = store.list_runs("sc_test")
        assert len(runs) == 1
        assert runs[0].schedule_id == "sc_test"

    def test_get_run(self, tmp_path):
        store = make_store(tmp_path)
        run = RunRecord(
            run_id="run_xyz",
            schedule_id="sc_test",
            occurrence_at_utc=datetime.utcnow().isoformat(),
        )
        store.append_run(run)

        loaded = store.get_run("run_xyz")
        assert loaded is not None
        assert loaded.run_id == "run_xyz"

    def test_recurring_item_roundtrip(self, tmp_path):
        store = make_store(tmp_path)
        item = ScheduleItem(
            id="sc_recurring",
            kind=ScheduleKind.TASK,
            title="Daily task",
            prompt="do something",
            schedule_type=ScheduleType.RECURRING,
            recurrence_rule=RecurrenceRule(kind="daily", time="09:00"),
            recurrence_text="every day at 09:00",
            delivery=Delivery(channel="telegram", chat_id="123"),
        )
        store.save(item)

        loaded = store.get("sc_recurring")
        assert loaded is not None
        assert loaded.schedule_type == ScheduleType.RECURRING
        assert loaded.recurrence_rule is not None
        assert loaded.recurrence_rule.kind == "daily"
        assert loaded.recurrence_rule.time == "09:00"

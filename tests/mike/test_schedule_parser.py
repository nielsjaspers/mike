from __future__ import annotations

from pathlib import Path

import pytest

from mike.scheduling.parser import Intent, ParsedScheduleIntent, ScheduleParser


class TestScheduleParser:
    def setup_method(self):
        self.parser = ScheduleParser()

    def test_list_intent(self):
        result = self.parser.parse("list")
        assert result.intent == Intent.LIST

    def test_list_alias(self):
        result = self.parser.parse("ls")
        assert result.intent == Intent.LIST

    def test_show_intent(self):
        result = self.parser.parse("show abc123")
        assert result.intent == Intent.SHOW
        assert result.schedule_id == "abc123"

    def test_delete_intent(self):
        result = self.parser.parse("delete abc123")
        assert result.intent == Intent.DELETE
        assert result.schedule_id == "abc123"

    def test_pause_intent(self):
        result = self.parser.parse("pause sc_123456")
        assert result.intent == Intent.PAUSE
        assert result.schedule_id == "sc_123456"

    def test_resume_intent(self):
        result = self.parser.parse("resume sc_123456")
        assert result.intent == Intent.RESUME
        assert result.schedule_id == "sc_123456"

    def test_run_now_intent(self):
        result = self.parser.parse("run sc_123456")
        assert result.intent == Intent.RUN_NOW
        assert result.schedule_id == "sc_123456"

    def test_help_intent(self):
        result = self.parser.parse("")
        assert result.intent == Intent.HELP

        result2 = self.parser.parse("help")
        assert result2.intent == Intent.HELP

    def test_add_daily_recurring(self):
        result = self.parser.parse("add every day at 09:00, remind me to stretch")
        assert result.intent == Intent.ADD
        assert result.is_recurring is True
        assert "remind me to stretch" in result.prompt
        assert result.recurrence_text is not None

    def test_add_weekday_recurring(self):
        result = self.parser.parse("add every weekday at 08:30, create daily plan")
        assert result.intent == Intent.ADD
        assert result.is_recurring is True
        assert "create daily plan" in result.prompt

    def test_add_monday_recurring(self):
        result = self.parser.parse("add every monday at 10:00, review sprint board")
        assert result.intent == Intent.ADD
        assert result.is_recurring is True

    def test_add_hourly_recurring(self):
        result = self.parser.parse("add every 2 hours, check CI status")
        assert result.intent == Intent.ADD
        assert result.is_recurring is True

    def test_add_relative_minutes(self):
        result = self.parser.parse("add in 5 minutes, remind me to stretch")
        assert result.intent == Intent.ADD
        assert result.is_recurring is False
        assert result.time_utc is not None

    def test_add_relative_hours(self):
        result = self.parser.parse("add in 3 hours, run report")
        assert result.intent == Intent.ADD
        assert result.is_recurring is False
        assert result.time_utc is not None

    def test_add_relative_days(self):
        result = self.parser.parse("add in 1 day, check in")
        assert result.intent == Intent.ADD
        assert result.is_recurring is False

    def test_add_tomorrow_at(self):
        result = self.parser.parse("add tomorrow at 09:00, morning routine")
        assert result.intent == Intent.ADD
        assert result.is_recurring is False

    def test_add_date_time(self):
        result = self.parser.parse("add 2026-04-01 14:30, quarterly review")
        assert result.intent == Intent.ADD
        assert result.is_recurring is False

    def test_natural_add_no_verb(self):
        result = self.parser.parse("in 5 hours, remind me to stretch")
        assert result.intent == Intent.ADD

    def test_update_intent(self):
        result = self.parser.parse("update sc_abc123 every day at 10:00, new prompt")
        assert result.intent == Intent.UPDATE
        assert result.schedule_id == "sc_abc123"
        assert result.raw_text is not None

    def test_format_help_contains_schedule(self):
        help_text = self.parser.format_help()
        assert "/schedule" in help_text
        assert "list" in help_text
        assert "add" in help_text

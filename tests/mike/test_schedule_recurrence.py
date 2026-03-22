from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mike.scheduling.recurrence import (
    NextRunCalculator,
    RecurrenceParser,
    parse_natural_datetime,
)


class TestRecurrenceParser:
    def setup_method(self):
        self.parser = RecurrenceParser()

    def test_parse_daily(self):
        rule = self.parser.parse("every day at 09:00")
        assert rule is not None
        assert rule.kind == "daily"
        assert rule.time == "09:00"

    def test_parse_weekday(self):
        rule = self.parser.parse("every weekday at 08:30")
        assert rule is not None
        assert rule.kind == "weekly"
        assert rule.weekdays == [0, 1, 2, 3, 4]
        assert rule.time == "08:30"

    def test_parse_monday(self):
        rule = self.parser.parse("every monday at 10:00")
        assert rule is not None
        assert rule.kind == "weekly"
        assert rule.weekdays == [0]
        assert rule.time == "10:00"

    def test_parse_tuesday(self):
        rule = self.parser.parse("every tuesday at 14:30")
        assert rule is not None
        assert rule.kind == "weekly"
        assert rule.weekdays == [1]

    def test_parse_hourly(self):
        rule = self.parser.parse("every 2 hours")
        assert rule is not None
        assert rule.kind == "interval"
        assert rule.interval_hours == 2

    def test_parse_simple_hourly(self):
        rule = self.parser.parse("every hour")
        assert rule is not None
        assert rule.kind == "interval"
        assert rule.interval_hours == 1

    def test_is_recurring_daily(self):
        assert self.parser.is_recurring("every day at 09:00") is True

    def test_is_recurring_weekday(self):
        assert self.parser.is_recurring("every weekday at 08:30") is True

    def test_is_recurring_monday(self):
        assert self.parser.is_recurring("every monday at 10:00") is True

    def test_is_recurring_hourly(self):
        assert self.parser.is_recurring("every 2 hours") is True

    def test_not_recurring(self):
        assert self.parser.is_recurring("in 5 minutes, do something") is False

    def test_parse_invalid_returns_none(self):
        assert self.parser.parse("some random text") is None


class TestNextRunCalculator:
    def setup_method(self):
        self.calc = NextRunCalculator("Europe/Amsterdam")

    def test_next_daily_returns_future(self):
        rule = RecurrenceParser().parse("every day at 09:00")
        assert rule is not None
        next_run = self.calc.next_run_utc(rule)
        assert next_run > datetime.utcnow()

    def test_next_weekly_returns_future(self):
        rule = RecurrenceParser().parse("every monday at 10:00")
        assert rule is not None
        next_run = self.calc.next_run_utc(rule)
        assert next_run > datetime.utcnow()

    def test_next_interval_hours(self):
        rule = RecurrenceParser().parse("every 1 hour")
        assert rule is not None
        next_run = self.calc.next_run_utc(rule)
        assert next_run > datetime.utcnow()
        delta = next_run - datetime.utcnow()
        assert timedelta(minutes=55) < delta < timedelta(hours=2)


class TestParseNaturalDatetime:
    def test_relative_minutes(self):
        result = parse_natural_datetime("in 5 minutes")
        assert result is not None
        assert result > datetime.utcnow()
        delta = result - datetime.utcnow()
        assert timedelta(minutes=4) < delta < timedelta(minutes=6)

    def test_relative_hours(self):
        result = parse_natural_datetime("in 3 hours")
        assert result is not None
        delta = result - datetime.utcnow()
        assert timedelta(hours=2, minutes=55) < delta < timedelta(hours=3, minutes=5)

    def test_relative_days(self):
        result = parse_natural_datetime("in 1 day")
        assert result is not None
        delta = result - datetime.utcnow()
        assert timedelta(days=0, hours=23) < delta < timedelta(days=1, hours=1)

    def test_tomorrow_at(self):
        result = parse_natural_datetime("tomorrow at 09:00")
        assert result is not None
        assert result.minute == 0
        from zoneinfo import ZoneInfo

        local = result.astimezone(ZoneInfo("Europe/Amsterdam"))
        assert local.hour == 9

    def test_absolute_date(self):
        result = parse_natural_datetime("2026-12-25 14:30")
        assert result is not None
        assert result.year == 2026
        assert result.month == 12
        assert result.day == 25
        assert result.hour == 14
        assert result.minute == 30

    def test_invalid_returns_none(self):
        assert parse_natural_datetime("whenever you feel like it") is None
        assert parse_natural_datetime("maybe tomorrow") is None

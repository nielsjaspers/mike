"""Recurrence parsing and next-run calculation for Mike scheduling."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mike.scheduling.types import RecurrenceRule

WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

TimeSpec = tuple[int, int]


def _parse_time(time_str: str) -> TimeSpec | None:
    match = re.match(r"^(\d{1,2}):(\d{2})$", time_str.strip())
    if not match:
        return None
    h, m = int(match.group(1)), int(match.group(2))
    if h > 23 or m > 59:
        return None
    return (h, m)


def _now_in_zone(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz))


def _utc_now() -> datetime:
    return datetime.utcnow()


class RecurrenceParser:
    WEEKDAY_RE = re.compile(
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2}:\d{2})",
        re.IGNORECASE,
    )
    WEEKDAYS_RE = re.compile(
        r"every\s+weekday\s+at\s+(\d{1,2}:\d{2})",
        re.IGNORECASE,
    )
    DAILY_RE = re.compile(
        r"every\s+day\s+at\s+(\d{1,2}:\d{2})",
        re.IGNORECASE,
    )
    HOURLY_RE = re.compile(
        r"every\s+(\d+)?\s*hours?",
        re.IGNORECASE,
    )
    TIME_RE = re.compile(r"at\s+(\d{1,2}:\d{2})", re.IGNORECASE)

    def parse(self, text: str) -> RecurrenceRule | None:
        text = text.strip()

        m = self.WEEKDAY_RE.search(text)
        if m:
            day_name = m.group(1).lower()
            time_str = m.group(2)
            return RecurrenceRule(
                kind="weekly",
                weekdays=[WEEKDAY_NAMES[day_name]],
                time=time_str,
            )

        m = self.WEEKDAYS_RE.search(text)
        if m:
            time_str = m.group(1)
            return RecurrenceRule(
                kind="weekly",
                weekdays=[0, 1, 2, 3, 4],
                time=time_str,
            )

        m = self.DAILY_RE.search(text)
        if m:
            time_str = m.group(1)
            return RecurrenceRule(
                kind="daily",
                time=time_str,
            )

        m = self.HOURLY_RE.search(text)
        if m:
            interval_str = m.group(1)
            interval = int(interval_str) if interval_str else 1
            return RecurrenceRule(
                kind="interval",
                interval_hours=interval,
                interval_count=interval,
            )

        return None

    def is_recurring(self, text: str) -> bool:
        return bool(
            self.WEEKDAY_RE.search(text)
            or self.WEEKDAYS_RE.search(text)
            or self.DAILY_RE.search(text)
            or self.HOURLY_RE.search(text)
        )


class NextRunCalculator:
    def __init__(self, timezone: str = "Europe/Amsterdam"):
        self.timezone = timezone

    def next_run_utc(self, rule: RecurrenceRule, after_utc: datetime | None = None) -> datetime:
        if rule.kind == "daily":
            return self._next_daily(rule.time, after_utc)
        elif rule.kind == "weekly":
            return self._next_weekly(rule.weekdays, rule.time, after_utc)
        elif rule.kind == "interval":
            return self._next_interval(rule.interval_hours or 1, after_utc)
        return _utc_now() + timedelta(minutes=5)

    def _next_daily(self, time_str: str, after_utc: datetime | None) -> datetime:
        tm = _parse_time(time_str)
        if tm is None:
            tm = (9, 0)

        local_now = _now_in_zone(self.timezone)
        target_local = local_now.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)

        if target_local <= local_now:
            target_local += timedelta(days=1)

        target_utc = target_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        return target_utc

    def _next_weekly(
        self, weekdays: list[int], time_str: str, after_utc: datetime | None
    ) -> datetime:
        tm = _parse_time(time_str)
        if tm is None:
            tm = (9, 0)

        local_now = _now_in_zone(self.timezone)
        current_weekday = local_now.weekday()

        for offset in range(8):
            day = (current_weekday + offset) % 7
            if day not in weekdays:
                continue
            candidate = local_now.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)
            if offset == 0:
                if candidate <= local_now:
                    continue
            candidate = candidate + timedelta(days=offset)
            if candidate.weekday() not in weekdays:
                continue
            target_utc = candidate.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            return target_utc

        return _utc_now() + timedelta(days=1)

    def _next_interval(self, hours: int, after_utc: datetime | None) -> datetime:
        after = after_utc or _utc_now()
        return after + timedelta(hours=hours)


def parse_natural_datetime(text: str) -> datetime | None:
    text = text.strip().lower()

    comma_idx = text.find(",")
    if comma_idx >= 0:
        time_part = text[:comma_idx].strip()
    else:
        time_part = text.split(maxsplit=3)[0] if len(text.split()) > 3 else text

    m = re.match(r"^in\s+(\d+)\s+(minutes?|mins?)(?:\s|,|$)", time_part)
    if m:
        mins = int(m.group(1))
        return _utc_now() + timedelta(minutes=mins)

    m = re.match(r"^in\s+(\d+)\s+(hours?|hrs?)(?:\s|,|$)", time_part)
    if m:
        hrs = int(m.group(1))
        return _utc_now() + timedelta(hours=hrs)

    m = re.match(r"^in\s+(\d+)\s+(days?)(?:\s|,|$)", time_part)
    if m:
        days = int(m.group(1))
        return _utc_now() + timedelta(days=days)

    m = re.match(r"^tomorrow\s+at\s+(\d{1,2}:\d{2})(?:\s|,|$)", time_part)
    if m:
        time_str = m.group(1)
        tm = _parse_time(time_str)
        if tm is None:
            return None
        local_tomorrow = _now_in_zone("Europe/Amsterdam") + timedelta(days=1)
        target = local_tomorrow.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)
        return target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})(?:\s|,|$)", time_part)
    if m:
        date_str, time_str = m.group(1), m.group(2)
        tm = _parse_time(time_str)
        if tm is None:
            return None
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d")
            target = target.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)
            target_utc = target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            return target_utc
        except ValueError:
            return None

    m = re.match(r"^tomorrow\s+at\s+(\d{1,2}:\d{2})$", text)
    if m:
        time_str = m.group(1)
        tm = _parse_time(time_str)
        if tm is None:
            return None
        local_tomorrow = _now_in_zone("Europe/Amsterdam") + timedelta(days=1)
        target = local_tomorrow.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)
        return target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})$", text)
    if m:
        date_str, time_str = m.group(1), m.group(2)
        tm = _parse_time(time_str)
        if tm is None:
            return None
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d")
            target = target.replace(hour=tm[0], minute=tm[1], second=0, microsecond=0)
            target_utc = target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            return target_utc
        except ValueError:
            return None

    return None

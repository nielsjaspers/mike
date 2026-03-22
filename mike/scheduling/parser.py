"""Command parser for /schedule command."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mike.scheduling.recurrence import NextRunCalculator, RecurrenceParser, parse_natural_datetime


class Intent(str, Enum):
    LIST = "list"
    SHOW = "show"
    ADD = "add"
    UPDATE = "update"
    PAUSE = "pause"
    RESUME = "resume"
    DELETE = "delete"
    RUN_NOW = "run_now"
    HELP = "help"


@dataclass
class ParsedScheduleIntent:
    intent: Intent
    schedule_id: str | None = None
    raw_text: str = ""
    recurrence_text: str | None = None
    time_utc: str | None = None
    prompt: str = ""
    is_recurring: bool = False
    error: str | None = None


class ScheduleParser:
    MANAGEMENT_VERBS = {
        "list",
        "ls",
        "show",
        "add",
        "update",
        "pause",
        "resume",
        "delete",
        "del",
        "rm",
        "run",
    }

    def parse(self, text: str) -> ParsedScheduleIntent:
        text = text.strip()
        if not text:
            return ParsedScheduleIntent(intent=Intent.HELP)

        lower = text.lower().strip()

        if lower in ("help", "-h", "--help"):
            return ParsedScheduleIntent(intent=Intent.HELP)

        if lower in ("list", "ls"):
            return ParsedScheduleIntent(intent=Intent.LIST)

        if lower.startswith("show "):
            sid = lower[5:].strip()
            return ParsedScheduleIntent(intent=Intent.SHOW, schedule_id=sid)

        if lower.startswith(("pause ", "delete ", "del ", "rm ", "resume ", "run ")):
            parts = lower.split(maxsplit=2)
            verb = parts[0]
            if len(parts) < 2:
                return ParsedScheduleIntent(
                    intent=Intent.HELP, error=f"Usage: /schedule {verb} <id>"
                )
            sid = parts[1].strip()
            if verb in ("pause",):
                return ParsedScheduleIntent(intent=Intent.PAUSE, schedule_id=sid)
            if verb in ("resume",):
                return ParsedScheduleIntent(intent=Intent.RESUME, schedule_id=sid)
            if verb in ("delete", "del", "rm"):
                return ParsedScheduleIntent(intent=Intent.DELETE, schedule_id=sid)
            if verb in ("run",):
                return ParsedScheduleIntent(intent=Intent.RUN_NOW, schedule_id=sid)

        if lower.startswith("update "):
            parts = text.split(maxsplit=2)
            if len(parts) < 3:
                return ParsedScheduleIntent(
                    intent=Intent.HELP, error="Usage: /schedule update <id> <spec>"
                )
            return ParsedScheduleIntent(
                intent=Intent.UPDATE,
                schedule_id=parts[1].strip(),
                raw_text=parts[2].strip(),
            )

        if lower.startswith("add "):
            raw = text[4:].strip()
            return self._parse_create(raw)

        return self._parse_create(text)

    def _parse_create(self, raw: str) -> ParsedScheduleIntent:
        recurrence_parser = RecurrenceParser()
        recurrence_calc = NextRunCalculator()

        is_recurring = recurrence_parser.is_recurring(raw)

        if is_recurring:
            rule = recurrence_parser.parse(raw)
            if rule is None:
                return ParsedScheduleIntent(
                    intent=Intent.ADD,
                    error="Could not parse recurrence pattern.",
                )
            comma_idx = raw.rfind(",")
            if comma_idx >= 0:
                prompt = raw[comma_idx + 1 :].strip()
            else:
                prompt = raw
            next_utc = recurrence_calc.next_run_utc(rule)
            return ParsedScheduleIntent(
                intent=Intent.ADD,
                raw_text=raw,
                recurrence_text=raw,
                time_utc=next_utc.isoformat(),
                prompt=prompt,
                is_recurring=True,
            )

        parsed_dt = parse_natural_datetime(raw)
        if parsed_dt is not None:
            comma_idx = raw.rfind(",")
            if comma_idx >= 0:
                prompt = raw[comma_idx + 1 :].strip()
            else:
                prompt = raw
            return ParsedScheduleIntent(
                intent=Intent.ADD,
                raw_text=raw,
                time_utc=parsed_dt.isoformat(),
                prompt=prompt,
                is_recurring=False,
            )

        return ParsedScheduleIntent(
            intent=Intent.ADD,
            error="Could not parse schedule time. Try 'in 5 hours, ...', 'tomorrow at 09:00, ...', or 'every day at 09:00, ...'",
        )

    def format_help(self) -> str:
        return (
            "/schedule - Show this help\n"
            "/schedule list - List all schedules\n"
            "/schedule show <id> - Show schedule details\n"
            "/schedule add <spec>, <prompt> - Create a schedule\n"
            "/schedule update <id> <spec> - Update a schedule\n"
            "/schedule pause <id> - Pause a schedule\n"
            "/schedule resume <id> - Resume a schedule\n"
            "/schedule delete <id> - Delete a schedule\n"
            "/schedule run <id> - Run a schedule immediately\n"
            "\nTime specs:\n"
            "  in N minutes/hours/days - relative time\n"
            "  tomorrow at HH:MM - specific date/time\n"
            "  YYYY-MM-DD HH:MM - absolute date/time\n"
            "Recurrence:\n"
            "  every day at HH:MM\n"
            "  every weekday at HH:MM\n"
            "  every monday at HH:MM\n"
            "  every N hours\n"
        )

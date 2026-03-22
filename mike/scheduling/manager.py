"""Schedule manager with polling loop, catch-up, and dispatch for Mike."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from loguru import logger

from mike.bus import MessageBus
from mike.config import MikeConfig
from mike.scheduling.parser import Intent, ScheduleParser
from mike.scheduling.recurrence import NextRunCalculator, RecurrenceParser
from mike.scheduling.store import ScheduleStore
from mike.scheduling.types import (
    Delivery,
    Execution,
    RunRecord,
    RunStatus,
    ScheduleItem,
    ScheduleKind,
    ScheduleType,
)
from mike.types import OutboundMessage

ExecuteCallback = Callable[
    [str, str, str, str, str | None],
    Awaitable[tuple[str, list[str]]],
]


class ScheduleManager:
    def __init__(
        self,
        config: MikeConfig,
        bus: MessageBus,
        store: ScheduleStore,
    ):
        self.config = config
        self.bus = bus
        self.store = store
        self._running = False
        self._task: asyncio.Task | None = None
        self._execute_callback: ExecuteCallback | None = None
        self._parser = ScheduleParser()
        self._recurrence_parser = RecurrenceParser()
        self._recurrence_calc = NextRunCalculator(config.schedule_timezone)

    def set_execute_callback(self, cb: ExecuteCallback) -> None:
        self._execute_callback = cb

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.current_task()
        try:
            await self._catch_up()
            await self._timer_loop()
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False
            self._task = None

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _timer_loop(self) -> None:
        while self._running:
            try:
                await self._process_due()
            except Exception:
                logger.exception("Schedule tick failed")
            await asyncio.sleep(self.config.schedule_poll_interval)

    async def _catch_up(self) -> None:
        now_utc = datetime.utcnow()
        due_items = []
        for item in self.store.list():
            if not item.enabled:
                continue
            if item.next_run_at_utc is None:
                continue
            try:
                next_run = datetime.fromisoformat(item.next_run_at_utc)
                if next_run <= now_utc:
                    due_items.append((item, next_run))
            except Exception:
                continue

        due_items.sort(key=lambda x: x[1])
        for item, _ in due_items[: self.config.schedule_max_catch_up]:
            self._dispatch_run(item, item.next_run_at_utc)

    async def _process_due(self) -> None:
        now_utc = datetime.utcnow()
        for item in self.store.list():
            if not item.enabled:
                continue
            if item.next_run_at_utc is None:
                continue
            try:
                next_run = datetime.fromisoformat(item.next_run_at_utc)
                if next_run <= now_utc:
                    await self._dispatch_and_advance(item)
            except Exception:
                logger.exception("Failed to process schedule {}", item.id)

    async def _dispatch_and_advance(self, item: ScheduleItem) -> None:
        occurrence_utc = item.next_run_at_utc
        if occurrence_utc is None:
            return

        self._dispatch_run(item, occurrence_utc)

        if item.schedule_type == ScheduleType.RECURRING and item.recurrence_rule:
            next_utc = self._recurrence_calc.next_run_utc(
                item.recurrence_rule, datetime.fromisoformat(occurrence_utc)
            )
            item.next_run_at_utc = next_utc.isoformat()
            item.last_run_at_utc = occurrence_utc
            self.store.save(item)
        elif item.schedule_type == ScheduleType.ONCE:
            item.enabled = False
            item.last_run_at_utc = occurrence_utc
            self.store.save(item)

    def _dispatch_run(self, item: ScheduleItem, occurrence_utc: str) -> None:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        idempotency_key = f"{item.id}:{occurrence_utc}"

        if self.store.has_succeeded_run(item.id, occurrence_utc):
            logger.debug("Skipping duplicate run {} for schedule {}", run_id, item.id)
            return

        run = RunRecord(
            run_id=run_id,
            schedule_id=item.id,
            occurrence_at_utc=occurrence_utc,
            attempt=1,
            status=RunStatus.RUNNING,
            idempotency_key=idempotency_key,
        )
        self.store.append_run(run)

        asyncio.create_task(self._execute_run(item, run))

    async def _execute_run(self, item: ScheduleItem, run: RunRecord) -> None:
        try:
            if item.kind == ScheduleKind.REMINDER:
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=item.delivery.channel,
                        chat_id=item.delivery.chat_id,
                        content=item.prompt,
                        media=[],
                        metadata={"message_thread_id": item.delivery.message_thread_id},
                    )
                )
                run.status = RunStatus.SUCCEEDED
                run.finished_at = datetime.utcnow().isoformat()
                run.result_summary = item.prompt[:500] if item.prompt else None
                self.store.append_run(run)
                return

            if self._execute_callback is None:
                logger.error("No execute callback set for schedule {}", item.id)
                return

            result_text, explicit_outputs = await self._execute_callback(
                prompt=item.prompt,
                chat_id=item.delivery.chat_id,
                schedule_id=item.id,
                run_id=run.run_id,
                model=item.execution.model,
            )

            run.status = RunStatus.SUCCEEDED
            run.finished_at = datetime.utcnow().isoformat()
            run.result_summary = result_text[:500] if result_text else None
            run.explicit_outputs = explicit_outputs
            self.store.append_run(run)

            await self._deliver_result(item, result_text, explicit_outputs)

        except asyncio.CancelledError:
            run.status = RunStatus.FAILED
            run.error = "Cancelled"
            run.finished_at = datetime.utcnow().isoformat()
            self.store.append_run(run)
            raise
        except Exception as exc:
            logger.exception("Schedule {} run {} failed", item.id, run.run_id)
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.finished_at = datetime.utcnow().isoformat()
            self.store.append_run(run)
            await self._handle_failure(item, run, exc)

    async def _handle_failure(self, item: ScheduleItem, run: RunRecord, exc: Exception) -> None:
        if run.attempt < self.config.schedule_retry_attempts:
            delay = self.config.schedule_retry_base_delay * (2 ** (run.attempt - 1))
            await asyncio.sleep(delay)
            new_run = RunRecord(
                run_id=f"run_{uuid.uuid4().hex[:8]}",
                schedule_id=item.id,
                occurrence_at_utc=run.occurrence_at_utc,
                attempt=run.attempt + 1,
                status=RunStatus.RETRY_SCHEDULED,
                idempotency_key=run.idempotency_key,
            )
            self.store.append_run(new_run)
            asyncio.create_task(self._execute_run(item, new_run))
        else:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=item.delivery.channel,
                    chat_id=item.delivery.chat_id,
                    content=(
                        f"[{item.id}] scheduled task failed after "
                        f"{self.config.schedule_retry_attempts} attempts: {exc}"
                    ),
                )
            )

    async def _deliver_result(
        self, item: ScheduleItem, result_text: str, explicit_outputs: list[str]
    ) -> None:
        if not result_text and not explicit_outputs:
            return

        summary, outputs = self._parse_output_block(result_text, explicit_outputs)

        content = f"[{item.id}] "
        if item.kind == ScheduleKind.REMINDER:
            content += summary if summary else result_text[:1000]
        else:
            content += summary if summary else (result_text or "Task completed.")[:1000]

        await self.bus.publish_outbound(
            OutboundMessage(
                channel=item.delivery.channel,
                chat_id=item.delivery.chat_id,
                content=content,
                media=outputs if item.execution.explicit_outputs_only else [],
                metadata={"message_thread_id": item.delivery.message_thread_id},
            )
        )

    def _parse_output_block(
        self, text: str, declared_outputs: list[str]
    ) -> tuple[str | None, list[str]]:
        summary = text
        outputs: list[str] = []

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                summary = data.get("summary", text)
                raw_outputs = data.get("outputs", [])
                if isinstance(raw_outputs, list):
                    for p in raw_outputs:
                        if isinstance(p, str) and p.strip():
                            outputs.append(p.strip())
            except Exception:
                pass

        for path in declared_outputs:
            if path not in outputs:
                outputs.append(path)

        return summary, outputs

    def handle_command(
        self,
        text: str,
        channel: str,
        chat_id: str,
    ) -> str:
        parsed = self._parser.parse(text)

        if parsed.error:
            return parsed.error

        if parsed.intent == Intent.HELP:
            return self._parser.format_help()

        if parsed.intent == Intent.LIST:
            return self._format_list()

        if parsed.intent == Intent.SHOW:
            return self._format_show(parsed.schedule_id)

        if parsed.intent == Intent.DELETE:
            return self._do_delete(parsed.schedule_id)

        if parsed.intent == Intent.PAUSE:
            return self._do_pause(parsed.schedule_id)

        if parsed.intent == Intent.RESUME:
            return self._do_resume(parsed.schedule_id)

        if parsed.intent == Intent.RUN_NOW:
            return self._do_run_now(parsed.schedule_id, channel, chat_id)

        if parsed.intent == Intent.UPDATE:
            return self._do_update(parsed.schedule_id, parsed.raw_text)

        if parsed.intent == Intent.ADD:
            return self._do_add(
                parsed.prompt,
                parsed.recurrence_text,
                parsed.time_utc,
                parsed.is_recurring,
                channel,
                chat_id,
            )

        return self._parser.format_help()

    def _format_list(self) -> str:
        items = self.store.list()
        if not items:
            return "No schedules. Use /schedule add <spec>, <prompt> to create one."
        lines = ["Schedules:"]
        for item in items:
            status = "active" if item.enabled else "paused"
            next_run = item.next_run_at_utc
            next_str = ""
            if next_run:
                try:
                    next_dt = datetime.fromisoformat(next_run)
                    next_str = f" next={next_dt.strftime('%Y-%m-%d %H:%M')}Z"
                except Exception:
                    next_str = f" next={next_run}"
            lines.append(
                f"- {item.id} [{item.kind.value}] [{status}]{next_str} {item.title or item.prompt[:40]}"
            )
        return "\n".join(lines)

    def _format_show(self, schedule_id: str | None) -> str:
        if not schedule_id:
            return "Usage: /schedule show <id>"
        item = self.store.get(schedule_id)
        if not item:
            return f"Schedule not found: {schedule_id}"
        next_run = item.next_run_at_utc or "none"
        last_run = item.last_run_at_utc or "never"
        lines = [
            f"Schedule: {item.id}",
            f"Kind: {item.kind.value}",
            f"Title: {item.title or '(none)'}",
            f"Enabled: {item.enabled}",
            f"Type: {item.schedule_type.value}",
            f"Next run: {next_run}",
            f"Last run: {last_run}",
            f"Prompt: {item.prompt}",
        ]
        if item.recurrence_text:
            lines.append(f"Recurrence: {item.recurrence_text}")
        runs = self.store.list_runs(item.id, limit=5)
        if runs:
            lines.append("Recent runs:")
            for r in runs:
                lines.append(
                    f"  - {r.run_id} [{r.status.value}] attempt={r.attempt} "
                    f"started={r.started_at[:19]}"
                )
        return "\n".join(lines)

    def _do_delete(self, schedule_id: str | None) -> str:
        if not schedule_id:
            return "Usage: /schedule delete <id>"
        item = self.store.get(schedule_id)
        if not item:
            return f"Schedule not found: {schedule_id}"
        self.store.delete(schedule_id)
        return f"Deleted schedule {schedule_id}."

    def _do_pause(self, schedule_id: str | None) -> str:
        if not schedule_id:
            return "Usage: /schedule pause <id>"
        item = self.store.get(schedule_id)
        if not item:
            return f"Schedule not found: {schedule_id}"
        item.enabled = False
        self.store.save(item)
        return f"Paused schedule {schedule_id}."

    def _do_resume(self, schedule_id: str | None) -> str:
        if not schedule_id:
            return "Usage: /schedule resume <id>"
        item = self.store.get(schedule_id)
        if not item:
            return f"Schedule not found: {schedule_id}"
        if item.next_run_at_utc is None and item.schedule_type == ScheduleType.RECURRING:
            if item.recurrence_rule:
                next_utc = self._recurrence_calc.next_run_utc(item.recurrence_rule)
                item.next_run_at_utc = next_utc.isoformat()
        item.enabled = True
        self.store.save(item)
        return f"Resumed schedule {schedule_id}."

    def _do_run_now(self, schedule_id: str | None, channel: str, chat_id: str) -> str:
        if not schedule_id:
            return "Usage: /schedule run <id>"
        item = self.store.get(schedule_id)
        if not item:
            return f"Schedule not found: {schedule_id}"
        item.delivery.channel = channel
        item.delivery.chat_id = chat_id
        self.store.save(item)
        now_utc = datetime.utcnow().isoformat()
        self._dispatch_run(item, now_utc)
        return f"Triggered immediate run of schedule {schedule_id}."

    def _do_update(self, schedule_id: str | None, raw_text: str) -> str:
        if not schedule_id:
            return "Usage: /schedule update <id> <spec>"
        item = self.store.get(schedule_id)
        if not item:
            return f"Schedule not found: {schedule_id}"
        if not raw_text:
            return "No update spec provided."

        recurrence_parser = RecurrenceParser()
        is_recurring = recurrence_parser.is_recurring(raw_text)
        parsed_dt = None
        try:
            from mike.scheduling.recurrence import parse_natural_datetime

            parsed_dt = parse_natural_datetime(raw_text)
        except Exception:
            pass

        if is_recurring:
            rule = recurrence_parser.parse(raw_text)
            if rule:
                item.schedule_type = ScheduleType.RECURRING
                item.recurrence_text = raw_text
                item.recurrence_rule = rule
                next_utc = self._recurrence_calc.next_run_utc(rule)
                item.next_run_at_utc = next_utc.isoformat()
                comma_idx = raw_text.rfind(",")
                if comma_idx >= 0:
                    item.prompt = raw_text[comma_idx + 1 :].strip()
        elif parsed_dt:
            item.schedule_type = ScheduleType.ONCE
            item.run_at_utc = parsed_dt.isoformat()
            item.next_run_at_utc = parsed_dt.isoformat()
            item.recurrence_text = None
            item.recurrence_rule = None
            comma_idx = raw_text.rfind(",")
            if comma_idx >= 0:
                item.prompt = raw_text[comma_idx + 1 :].strip()
        else:
            comma_idx = raw_text.rfind(",")
            if comma_idx >= 0:
                item.prompt = raw_text[comma_idx + 1 :].strip()

        self.store.save(item)
        return f"Updated schedule {schedule_id}."

    def _do_add(
        self,
        prompt: str,
        recurrence_text: str | None,
        time_utc: str | None,
        is_recurring: bool,
        channel: str,
        chat_id: str,
    ) -> str:
        if not prompt:
            return "No task/prompt specified."
        if not is_recurring and not time_utc:
            return "Could not determine schedule time."

        item_id = self.store.new_id()
        title = prompt[:40]

        if is_recurring:
            rule = self._recurrence_parser.parse(recurrence_text or prompt)
            next_utc = self._recurrence_calc.next_run_utc(rule) if rule else None
            item = ScheduleItem(
                id=item_id,
                kind=ScheduleKind.TASK,
                title=title,
                prompt=prompt,
                timezone=self.config.schedule_timezone,
                schedule_type=ScheduleType.RECURRING,
                recurrence_text=recurrence_text,
                recurrence_rule=rule,
                next_run_at_utc=next_utc.isoformat() if next_utc else None,
                delivery=Delivery(channel=channel, chat_id=chat_id),
                execution=Execution(),
            )
        else:
            item = ScheduleItem(
                id=item_id,
                kind=ScheduleKind.TASK,
                title=title,
                prompt=prompt,
                timezone=self.config.schedule_timezone,
                schedule_type=ScheduleType.ONCE,
                run_at_utc=time_utc,
                next_run_at_utc=time_utc,
                delivery=Delivery(channel=channel, chat_id=chat_id),
                execution=Execution(),
            )

        self.store.save(item)
        return f"Created schedule [{item_id}]. Next run: {item.next_run_at_utc or 'unknown'}"

    def tool_create(
        self,
        text: str,
        prompt: str | None,
        when_text: str | None,
        recurrence_text: str | None,
        kind_str: str | None,
        channel: str,
        chat_id: str,
    ) -> dict[str, Any]:
        try:
            from mike.scheduling.parser import ScheduleParser
            from mike.scheduling.recurrence import NextRunCalculator, RecurrenceParser

            parser = ScheduleParser()
            parsed = parser.parse(text or "")

            if parsed.error and not (when_text or recurrence_text):
                return {
                    "ok": False,
                    "action": "create",
                    "error_code": "invalid_time",
                    "message": parsed.error,
                    "needs_clarification": True,
                    "clarification_question": "I couldn't understand the time. Try: 'in 5 minutes, ...' or 'every day at 09:00, ...'",
                }

            effective_prompt = prompt if prompt else parsed.prompt
            if not effective_prompt and when_text:
                effective_prompt = when_text
            if not effective_prompt:
                return {
                    "ok": False,
                    "action": "create",
                    "error_code": "missing_field",
                    "message": "No task or reminder specified.",
                    "needs_clarification": True,
                    "clarification_question": "What should I remind you to do, or what task should run?",
                }

            is_recurring = parsed.is_recurring or (recurrence_text is not None)
            time_utc = parsed.time_utc
            rec_text = recurrence_text or parsed.recurrence_text
            rec_rule = None
            next_run = None

            if is_recurring:
                rec_parser = RecurrenceParser()
                rec_calc = NextRunCalculator(self.config.schedule_timezone)
                rec_rule = rec_parser.parse(recurrence_text or text)
                if rec_rule:
                    next_run = rec_calc.next_run_utc(rec_rule)
                else:
                    return {
                        "ok": False,
                        "action": "create",
                        "error_code": "invalid_recurrence",
                        "message": "Could not parse recurrence pattern.",
                        "needs_clarification": True,
                        "clarification_question": "Try: 'every day at 09:00', 'every weekday at 08:30', 'every monday at 10:00', or 'every 2 hours'",
                    }
            elif when_text:
                from mike.scheduling.recurrence import parse_natural_datetime

                dt = parse_natural_datetime(when_text)
                if dt:
                    time_utc = dt.isoformat()
                    next_run = dt
                else:
                    return {
                        "ok": False,
                        "action": "create",
                        "error_code": "invalid_time",
                        "message": f"Could not understand when: {when_text}",
                        "needs_clarification": True,
                        "clarification_question": "Try: 'in 5 minutes', 'tomorrow at 09:00', or '2026-04-01 14:30'",
                    }
            elif parsed.time_utc:
                next_run = datetime.fromisoformat(parsed.time_utc)

            if not next_run and not is_recurring:
                return {
                    "ok": False,
                    "action": "create",
                    "error_code": "invalid_time",
                    "message": "No time specified.",
                    "needs_clarification": True,
                    "clarification_question": "When should this run? Try: 'in 5 minutes', 'tomorrow at 09:00', or 'every day at 09:00'",
                }

            item_id = self.store.new_id()
            title = effective_prompt[:40]
            normalized_kind = (kind_str or "").strip().lower()
            if normalized_kind == "reminder":
                schedule_kind = ScheduleKind.REMINDER
            elif normalized_kind == "task":
                schedule_kind = ScheduleKind.TASK
            else:
                kind_hint_text = f"{text} {effective_prompt}".lower()
                schedule_kind = (
                    ScheduleKind.REMINDER
                    if ("remind" in kind_hint_text or "reminder" in kind_hint_text)
                    else ScheduleKind.TASK
                )

            if is_recurring:
                item = ScheduleItem(
                    id=item_id,
                    kind=schedule_kind,
                    title=title,
                    prompt=effective_prompt,
                    timezone=self.config.schedule_timezone,
                    schedule_type=ScheduleType.RECURRING,
                    recurrence_text=rec_text,
                    recurrence_rule=rec_rule,
                    next_run_at_utc=next_run.isoformat() if next_run else None,
                    delivery=Delivery(channel=channel, chat_id=chat_id),
                    execution=Execution(),
                )
            else:
                item = ScheduleItem(
                    id=item_id,
                    kind=schedule_kind,
                    title=title,
                    prompt=effective_prompt,
                    timezone=self.config.schedule_timezone,
                    schedule_type=ScheduleType.ONCE,
                    run_at_utc=time_utc,
                    next_run_at_utc=next_run.isoformat() if next_run else time_utc,
                    delivery=Delivery(channel=channel, chat_id=chat_id),
                    execution=Execution(),
                )

            self.store.save(item)

            local_tz = ZoneInfo(self.config.schedule_timezone)
            next_run_utc = (
                datetime.fromisoformat(item.next_run_at_utc) if item.next_run_at_utc else None
            )
            next_run_local = None
            if next_run_utc:
                next_run_local = next_run_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(local_tz)
                next_run_formatted = next_run_local.strftime("%Y-%m-%d %H:%M %Z")
            else:
                next_run_formatted = "unknown"

            return {
                "ok": True,
                "action": "create",
                "message": f"Created schedule [{item_id}]. Next run: {next_run_formatted}",
                "data": {
                    "schedule_id": item_id,
                    "kind": item.kind.value,
                    "title": title,
                    "prompt": effective_prompt,
                    "next_run_at_utc": item.next_run_at_utc,
                    "next_run_local": next_run_local.isoformat()
                    if next_run_local is not None
                    else None,
                    "schedule_type": item.schedule_type.value,
                },
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "create",
                "error_code": "internal_error",
                "message": f"Failed to create schedule: {exc}",
            }

    def tool_list(self) -> dict[str, Any]:
        try:
            items = self.store.list()
            return {
                "ok": True,
                "action": "list",
                "message": f"{len(items)} schedule(s)" if items else "No schedules found.",
                "data": {
                    "items": [
                        {
                            "id": item.id,
                            "kind": item.kind.value,
                            "title": item.title,
                            "prompt": item.prompt,
                            "enabled": item.enabled,
                            "schedule_type": item.schedule_type.value,
                            "next_run_at_utc": item.next_run_at_utc,
                            "last_run_at_utc": item.last_run_at_utc,
                        }
                        for item in items
                    ]
                },
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "list",
                "error_code": "internal_error",
                "message": f"Failed to list schedules: {exc}",
            }

    def tool_show(self, schedule_id: str | None) -> dict[str, Any]:
        try:
            if not schedule_id:
                return {
                    "ok": False,
                    "action": "show",
                    "error_code": "missing_field",
                    "message": "schedule_id is required for show",
                }
            item = self.store.get(schedule_id)
            if not item:
                return {
                    "ok": False,
                    "action": "show",
                    "error_code": "not_found",
                    "message": f"Schedule not found: {schedule_id}",
                }
            runs = self.store.list_runs(schedule_id, limit=10)
            return {
                "ok": True,
                "action": "show",
                "message": f"Schedule: {item.id}",
                "data": {
                    "id": item.id,
                    "kind": item.kind.value,
                    "title": item.title,
                    "prompt": item.prompt,
                    "enabled": item.enabled,
                    "schedule_type": item.schedule_type.value,
                    "next_run_at_utc": item.next_run_at_utc,
                    "last_run_at_utc": item.last_run_at_utc,
                    "recurrence_text": item.recurrence_text,
                    "delivery": {
                        "channel": item.delivery.channel,
                        "chat_id": item.delivery.chat_id,
                    },
                    "recent_runs": [
                        {
                            "run_id": r.run_id,
                            "status": r.status.value,
                            "attempt": r.attempt,
                            "started_at": r.started_at,
                            "finished_at": r.finished_at,
                            "error": r.error,
                        }
                        for r in runs
                    ],
                },
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "show",
                "error_code": "internal_error",
                "message": f"Failed to show schedule: {exc}",
            }

    def tool_update(self, schedule_id: str | None, text: str) -> dict[str, Any]:
        try:
            if not schedule_id:
                return {
                    "ok": False,
                    "action": "update",
                    "error_code": "missing_field",
                    "message": "schedule_id is required for update",
                }
            item = self.store.get(schedule_id)
            if not item:
                return {
                    "ok": False,
                    "action": "update",
                    "error_code": "not_found",
                    "message": f"Schedule not found: {schedule_id}",
                }
            from mike.scheduling.parser import ScheduleParser
            from mike.scheduling.recurrence import NextRunCalculator, RecurrenceParser

            parser = ScheduleParser()
            parsed = parser.parse(text)

            if parsed.error and not text.strip():
                return {
                    "ok": False,
                    "action": "update",
                    "error_code": "invalid_spec",
                    "message": "No update specification provided.",
                }

            if parsed.is_recurring:
                rec_parser = RecurrenceParser()
                rec_calc = NextRunCalculator(self.config.schedule_timezone)
                rule = rec_parser.parse(text)
                if rule:
                    item.schedule_type = ScheduleType.RECURRING
                    item.recurrence_text = parsed.recurrence_text or text
                    item.recurrence_rule = rule
                    next_utc = rec_calc.next_run_utc(rule)
                    item.next_run_at_utc = next_utc.isoformat()
                if parsed.prompt:
                    item.prompt = parsed.prompt
            elif parsed.time_utc:
                item.schedule_type = ScheduleType.ONCE
                item.run_at_utc = parsed.time_utc
                item.next_run_at_utc = parsed.time_utc
                item.recurrence_text = None
                item.recurrence_rule = None
                if parsed.prompt:
                    item.prompt = parsed.prompt
            else:
                if parsed.prompt:
                    item.prompt = parsed.prompt

            self.store.save(item)
            return {
                "ok": True,
                "action": "update",
                "message": f"Updated schedule {schedule_id}.",
                "data": {"id": schedule_id, "next_run_at_utc": item.next_run_at_utc},
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "update",
                "error_code": "internal_error",
                "message": f"Failed to update schedule: {exc}",
            }

    def tool_pause(self, schedule_id: str | None) -> dict[str, Any]:
        try:
            if not schedule_id:
                return {
                    "ok": False,
                    "action": "pause",
                    "error_code": "missing_field",
                    "message": "schedule_id is required for pause",
                }
            item = self.store.get(schedule_id)
            if not item:
                return {
                    "ok": False,
                    "action": "pause",
                    "error_code": "not_found",
                    "message": f"Schedule not found: {schedule_id}",
                }
            item.enabled = False
            self.store.save(item)
            return {
                "ok": True,
                "action": "pause",
                "message": f"Paused schedule {schedule_id}.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "pause",
                "error_code": "internal_error",
                "message": f"Failed to pause schedule: {exc}",
            }

    def tool_resume(self, schedule_id: str | None) -> dict[str, Any]:
        try:
            if not schedule_id:
                return {
                    "ok": False,
                    "action": "resume",
                    "error_code": "missing_field",
                    "message": "schedule_id is required for resume",
                }
            item = self.store.get(schedule_id)
            if not item:
                return {
                    "ok": False,
                    "action": "resume",
                    "error_code": "not_found",
                    "message": f"Schedule not found: {schedule_id}",
                }
            if item.next_run_at_utc is None and item.schedule_type == ScheduleType.RECURRING:
                if item.recurrence_rule:
                    rec_calc = NextRunCalculator(self.config.schedule_timezone)
                    next_utc = rec_calc.next_run_utc(item.recurrence_rule)
                    item.next_run_at_utc = next_utc.isoformat()
            item.enabled = True
            self.store.save(item)
            return {
                "ok": True,
                "action": "resume",
                "message": f"Resumed schedule {schedule_id}.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "resume",
                "error_code": "internal_error",
                "message": f"Failed to resume schedule: {exc}",
            }

    def tool_delete(self, schedule_id: str | None) -> dict[str, Any]:
        try:
            if not schedule_id:
                return {
                    "ok": False,
                    "action": "delete",
                    "error_code": "missing_field",
                    "message": "schedule_id is required for delete",
                }
            item = self.store.get(schedule_id)
            if not item:
                return {
                    "ok": False,
                    "action": "delete",
                    "error_code": "not_found",
                    "message": f"Schedule not found: {schedule_id}",
                }
            self.store.delete(schedule_id)
            return {
                "ok": True,
                "action": "delete",
                "message": f"Deleted schedule {schedule_id}.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "delete",
                "error_code": "internal_error",
                "message": f"Failed to delete schedule: {exc}",
            }

    def tool_run_now(self, schedule_id: str | None, channel: str, chat_id: str) -> dict[str, Any]:
        try:
            if not schedule_id:
                return {
                    "ok": False,
                    "action": "run_now",
                    "error_code": "missing_field",
                    "message": "schedule_id is required for run_now",
                }
            item = self.store.get(schedule_id)
            if not item:
                return {
                    "ok": False,
                    "action": "run_now",
                    "error_code": "not_found",
                    "message": f"Schedule not found: {schedule_id}",
                }
            item.delivery.channel = channel
            item.delivery.chat_id = chat_id
            self.store.save(item)
            now_utc = datetime.utcnow().isoformat()
            self._dispatch_run(item, now_utc)
            return {
                "ok": True,
                "action": "run_now",
                "message": f"Triggered immediate run of schedule {schedule_id}.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "run_now",
                "error_code": "internal_error",
                "message": f"Failed to run schedule: {exc}",
            }

    def tool_status(self) -> dict[str, Any]:
        try:
            items = self.store.list()
            active = [i for i in items if i.enabled and i.next_run_at_utc]
            paused = [i for i in items if not i.enabled]
            return {
                "ok": True,
                "action": "status",
                "message": f"Total: {len(items)}, Active: {len(active)}, Paused: {len(paused)}",
                "data": {
                    "total": len(items),
                    "active": len(active),
                    "paused": len(paused),
                    "upcoming": [
                        {"id": i.id, "next_run_at_utc": i.next_run_at_utc, "title": i.title}
                        for i in sorted(active, key=lambda x: x.next_run_at_utc or "")[:5]
                    ],
                },
            }
        except Exception as exc:
            return {
                "ok": False,
                "action": "status",
                "error_code": "internal_error",
                "message": f"Failed to get status: {exc}",
            }

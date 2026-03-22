"""Native schedule tool for Mike."""

from __future__ import annotations

import json
from typing import Any

from mike.tools.base import Tool


class ScheduleTool(Tool):
    def __init__(self, manager: Any = None):
        self._manager = manager
        self._channel = "cli"
        self._chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    def set_manager(self, manager: Any) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "schedule"

    @property
    def description(self) -> str:
        return (
            "Create, list, show, update, pause, resume, delete, or run scheduled tasks. "
            "Use for reminders and timed background tasks. "
            "Call this tool before responding to the user about any schedule operation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create",
                        "list",
                        "show",
                        "update",
                        "pause",
                        "resume",
                        "delete",
                        "run_now",
                        "status",
                        "help",
                    ],
                    "description": "The schedule action to perform",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Natural language description for create or update actions. "
                        "Examples: 'in 5 minutes remind me to stretch', "
                        "'every day at 09:00 give me a status update', "
                        "'tomorrow at 14:00 review the sprint board'"
                    ),
                },
                "schedule_id": {
                    "type": "string",
                    "description": "Schedule ID for show/pause/resume/delete/run_now actions",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task or reminder text",
                },
                "when_text": {
                    "type": "string",
                    "description": "When to run: 'in 5 minutes', 'tomorrow at 09:00', 'every day at 09:00', etc.",
                },
                "recurrence_text": {
                    "type": "string",
                    "description": "Recurrence pattern: 'every day at HH:MM', 'every weekday at HH:MM', 'every monday at HH:MM', 'every N hours'",
                },
                "kind": {
                    "type": "string",
                    "enum": ["reminder", "task"],
                    "description": "Kind of schedule: 'reminder' or 'task'",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        if self._manager is None:
            return json.dumps(
                {
                    "ok": False,
                    "action": action,
                    "error_code": "not_initialized",
                    "message": "Schedule manager not available.",
                }
            )
        text = kwargs.get("text", "")
        schedule_id = kwargs.get("schedule_id")
        prompt = kwargs.get("prompt")
        when_text = kwargs.get("when_text")
        recurrence_text = kwargs.get("recurrence_text")
        kind_str = kwargs.get("kind")

        if action == "create":
            return self._create(text, prompt, when_text, recurrence_text, kind_str)
        if action == "list":
            return self._list()
        if action == "show":
            return self._show(schedule_id)
        if action == "update":
            return self._update(schedule_id, text)
        if action == "pause":
            return self._pause(schedule_id)
        if action == "resume":
            return self._resume(schedule_id)
        if action == "delete":
            return self._delete(schedule_id)
        if action == "run_now":
            return self._run_now(schedule_id)
        if action == "status":
            return self._status()
        if action == "help":
            return self._help()
        return json.dumps(
            {
                "ok": False,
                "action": action,
                "error_code": "unsupported_action",
                "message": f"Unsupported action: {action}",
            }
        )

    def _create(
        self,
        text: str,
        prompt: str | None,
        when_text: str | None,
        recurrence_text: str | None,
        kind_str: str | None,
    ) -> str:
        result = self._manager.tool_create(
            text=text,
            prompt=prompt,
            when_text=when_text,
            recurrence_text=recurrence_text,
            kind_str=kind_str,
            channel=self._channel,
            chat_id=self._chat_id,
        )
        return json.dumps(result)

    def _list(self) -> str:
        result = self._manager.tool_list()
        return json.dumps(result)

    def _show(self, schedule_id: str | None) -> str:
        result = self._manager.tool_show(schedule_id)
        return json.dumps(result)

    def _update(self, schedule_id: str | None, text: str) -> str:
        result = self._manager.tool_update(schedule_id, text)
        return json.dumps(result)

    def _pause(self, schedule_id: str | None) -> str:
        result = self._manager.tool_pause(schedule_id)
        return json.dumps(result)

    def _resume(self, schedule_id: str | None) -> str:
        result = self._manager.tool_resume(schedule_id)
        return json.dumps(result)

    def _delete(self, schedule_id: str | None) -> str:
        result = self._manager.tool_delete(schedule_id)
        return json.dumps(result)

    def _run_now(self, schedule_id: str | None) -> str:
        result = self._manager.tool_run_now(schedule_id, self._channel, self._chat_id)
        return json.dumps(result)

    def _status(self) -> str:
        result = self._manager.tool_status()
        return json.dumps(result)

    def _help(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "action": "help",
                "message": (
                    "Schedule tool actions: create, list, show, update, pause, resume, delete, run_now, status, help\n"
                    "create: /schedule create text='in 5 minutes, remind me to stretch'\n"
                    "list: /schedule list\n"
                    "show: /schedule show schedule_id='sc_abc123'\n"
                    "pause/resume/delete/run_now: requires schedule_id\n"
                    "text field accepts natural language like 'in 5 minutes, remind me to stretch'"
                ),
            }
        )

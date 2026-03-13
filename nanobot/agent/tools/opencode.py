"""Tools backed by OpenCode for delegation and local POC runs."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class OpenCodeDelegateTool(Tool):
    """Delegate an arbitrary subtask to OpenCode Serve."""

    def __init__(self, manager: Any):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._model: str | None = None

    def set_context(self, channel: str, chat_id: str, model: str | None = None) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"
        self._model = model

    @property
    def name(self) -> str:
        return "opencode_delegate"

    @property
    def description(self) -> str:
        return (
            "Delegate a complex task to OpenCode Serve. Use this for web-heavy or autonomous subtasks where the "
            "OpenCode harness is a better fit than native Nanobot tools."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The delegated task to run via OpenCode Serve",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for status display",
                },
            },
            "required": ["task"],
        }

    async def execute(self, **kwargs: Any) -> str:
        task = kwargs.get("task")
        if not isinstance(task, str) or not task.strip():
            return "Error: Missing required parameter: task"
        label = kwargs.get("label")
        return await self._manager.run_local_task(
            task=task,
            label=label,
            session_key=self._session_key,
            channel=self._origin_channel,
            chat_id=self._origin_chat_id,
            model=self._model,
        )

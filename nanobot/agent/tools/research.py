"""Tool for OpenCode-backed research task lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.research.manager import ResearchManager


class ResearchTool(Tool):
    """Start and manage background OpenCode research tasks."""

    def __init__(self, manager: "ResearchManager"):
        self._manager = manager
        self._channel = "cli"
        self._chat_id = "direct"
        self._session_key = "cli:direct"
        self._model: str | None = None

    def set_context(self, channel: str, chat_id: str, model: str | None = None) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"
        self._model = model

    @property
    def name(self) -> str:
        return "research"

    @property
    def description(self) -> str:
        return (
            "Start or manage an OpenCode-backed background research task. "
            "Use this for questions that need asynchronous research, prototyping, or long-running work."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "status", "cancel"],
                    "description": "Research lifecycle action",
                },
                "query": {
                    "type": "string",
                    "description": "Research query when action=start",
                },
                "task_id": {
                    "type": "string",
                    "description": "Optional task id for cancel",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action")
        if not isinstance(action, str) or not action:
            return "Error: action is required"
        if action == "start":
            query = kwargs.get("query")
            if not isinstance(query, str) or not query.strip():
                return "Error: query is required when action=start"
            return await self._manager.start_task(
                query=query,
                session_key=self._session_key,
                channel=self._channel,
                chat_id=self._chat_id,
                model=self._model,
            )
        if action == "status":
            return self._manager.format_status(self._session_key)
        if action == "cancel":
            task_id = kwargs.get("task_id")
            if task_id:
                await self._manager.cancel_task(task_id)
                return f"Cancelled research task {task_id}."
            count = await self._manager.cancel_by_session(self._session_key)
            return (
                f"Cancelled {count} research task(s)."
                if count
                else "No running research task found for this chat."
            )
        return f"Error: Unsupported action '{action}'"

"""Tools backed by OpenCode Serve for search and delegation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class OpenCodeWebSearchTool(Tool):
    """Search the web by delegating to an OpenCode-backed background task."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using OpenCode Serve's built-in web search harness. "
            "This returns the actual search result text from OpenCode instead of using Brave Search."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {
                    "type": "integer",
                    "description": "Optional target number of results to consider",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: Missing required parameter: query"
        count = kwargs.get("count")
        return await self._manager.run_opencode_query(
            text=(
                "Use your built-in websearch tool to answer this query. Perform the search, inspect the results, "
                "and return concrete findings with sources. Do not describe what you plan to do.\n\n"
                f"Query: {query}\n"
                + (f"Preferred result count: {count}\n" if isinstance(count, int) else "")
            ),
            label=f"search: {query[:24]}",
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )


class OpenCodeDelegateTool(Tool):
    """Delegate an arbitrary subtask to OpenCode Serve."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

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
        return await self._manager.run_opencode_query(
            text=(
                "Execute this delegated task inside the OpenCode Serve harness. "
                "Prefer OpenCode's built-in tools when useful, especially for code changes, websearch, apply_patch, "
                f"edit, read, and bash.\n\n{task}"
            ),
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )

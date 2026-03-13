"""OpenCode Exa search tool wrapper."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urlparse

from nanobot.agent.tools.base import Tool


class OpencodeSearchTool(Tool):
    """Search the web via OpenCode's Exa tool."""

    @property
    def name(self) -> str:
        return "opencode_search"

    @property
    def description(self) -> str:
        return "Search the web via OpenCode Exa tool. Returns titles, URLs, snippets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {
                    "type": "integer",
                    "description": "Results (1-10)",
                    "minimum": 1,
                    "maximum": 10,
                },
                "attach_url": {"type": "string", "description": "Local opencode serve URL"},
            },
            "required": ["query"],
        }

    def __init__(
        self,
        cli_bin: str | None = None,
        timeout: int = 120,
        agent: str | None = None,
        provider_id: str | None = None,
    ):
        self.cli_bin = cli_bin or os.environ.get("OPENCODE_BIN", "opencode")
        self.timeout = timeout
        self.agent = agent
        self.provider_id = provider_id or os.environ.get(
            "OPENCODE_MODEL_PROVIDER_ID", "opencode-go"
        )
        self.model: str | None = None

    def set_context(self, channel: str, chat_id: str, model: str | None = None) -> None:
        del channel, chat_id
        self.model = model

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: Missing required parameter: query"
        count = kwargs.get("count")
        attach_url = kwargs.get("attach_url")
        if not _is_truthy(os.environ.get("OPENCODE_ENABLE_EXA", "1")):
            return "Error: OPENCODE_ENABLE_EXA is not enabled"

        attach_url = attach_url or os.environ.get("OPENCODE_ATTACH_URL", "http://127.0.0.1:4096")
        if attach_url and not _is_local_url(attach_url):
            return "Error: attach_url must be localhost (privacy policy)"

        n = min(max(count or 5, 1), 10)
        prompt = (
            "Use the Exa web search tool to search for: "
            f"{query!r}. Return ONLY JSON with this schema: "
            '{"results":[{"title":...,"url":...,"snippet":...}]}. '
            f"Limit to {n} results. No extra text."
        )

        cmd = [self.cli_bin, "run"]
        if attach_url:
            cmd.extend(["--attach", attach_url])
        if self.model:
            cmd.extend(["--model", f"{self.provider_id}/{self.model}"])
        if self.agent:
            cmd.extend(["--agent", self.agent])
        cmd.append(prompt)

        stdout, stderr, code = await self._run(cmd)
        if code != 0:
            err = stdout.strip() or stderr.strip() or "opencode run failed"
            return f"Error: {err}"

        # Try to extract JSON from combined output
        text = stdout.strip()
        data = _extract_json(text)
        if data is None:
            return text or "Error: empty response from opencode"

        return json.dumps(data, ensure_ascii=False)

    async def _run(self, cmd: list[str]) -> tuple[str, str, int | None]:
        try:
            # Use --format json to get structured output that we can parse
            if "--format" not in cmd:
                cmd.insert(2, "json")  # Insert after "run"
                cmd.insert(2, "--format")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                process.kill()
                return "", f"Timed out after {self.timeout} seconds", 124

            out = stdout.decode("utf-8", errors="replace") if stdout else ""
            err = stderr.decode("utf-8", errors="replace") if stderr else ""
            return out, err, process.returncode or 0
        except FileNotFoundError:
            return "", f"'{self.cli_bin}' not found. Install OpenCode or set OPENCODE_BIN.", 127
        except Exception as exc:
            return "", f"Error executing opencode: {exc}", 1


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    
    import re
    
    # Strip ANSI escape codes
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    
    # Parse JSON events from opencode --format json output
    # Look for tool_use events that contain web search results
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "tool_use":
                part = event.get("part", {})
                if part.get("tool") == "websearch" and part.get("state", {}).get("status") == "completed":
                    output = part.get("state", {}).get("output", "")
                    if output:
                        # Parse the text output to extract individual results
                        results.extend(_parse_websearch_output(output))
            elif "results" in event:
                # Direct JSON response
                return event
        except json.JSONDecodeError:
            continue
    
    if results:
        return {"results": results}
    
    # Fallback: try parsing the whole text as JSON
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    
    # Extract JSON from markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1).strip())
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    
    # Extract bare JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    
    return None


def _parse_websearch_output(output: str) -> list[dict[str, str]]:
    """Parse web search output text into structured results."""
    results = []
    import re
    
    # Split by "Title:" to get individual results
    # Each result has: Title, Published Date, URL, Text
    pattern = r"Title:\s*(.+?)\n(?:Published Date:\s*(.+?)\n)?(?:Author:\s*(.+?)\n)?URL:\s*(.+?)\nText:\s*(.+?)(?=\n\nTitle:|$)"
    matches = re.findall(pattern, output, re.DOTALL)
    
    for match in matches:
        title, published, author, url, text = match
        results.append({
            "title": title.strip(),
            "url": url.strip(),
            "snippet": text.strip()[:300] + "..." if len(text.strip()) > 300 else text.strip()
        })
    
    return results


def _is_truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _is_local_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}
    except Exception:
        return False

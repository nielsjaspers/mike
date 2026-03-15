"""Helpers for explicit HISTORY.md usage."""

from __future__ import annotations

import re
from pathlib import Path


def search_history(path: Path, query: str, limit: int = 20) -> list[str]:
    if not path.exists() or not query.strip():
        return []
    regex = re.compile(re.escape(query), re.IGNORECASE)
    matches: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if regex.search(line):
            matches.append(line)
        if len(matches) >= limit:
            break
    return matches

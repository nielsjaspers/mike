"""Prompt builders for Nocturne writing sessions."""

from __future__ import annotations

import json
from pathlib import Path

from .types import StoryState, WorkMetadata


def load_creative_soul(data_dir: Path) -> str:
    path = data_dir / "CREATIVE_SOUL.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_daily_prompt(
    piece_type: str,
    creative_soul: str,
    recent_works: list[WorkMetadata],
    directive: str | None = None,
) -> str:
    type_guidance = {
        "poetry": "Write a vivid, image-rich poem with a distinct rhythmic shape.",
        "philosophy": "Write a short philosophical essay rooted in concrete human experience.",
        "tech_speculation": "Write a grounded technology speculation anchored in real trends.",
        "fiction": "Write a character-driven short fiction scene with emotional stakes.",
    }.get(piece_type, "Write an original creative piece.")
    recent = ""
    if recent_works:
        lines = []
        for work in recent_works[:8]:
            lines.append(f"- [{work.piece_type}] {work.title}: {work.summary}")
        recent = "\n".join(lines)
    parts = [
        "You are in Nocturne mode. Produce one polished, original piece.",
        f"Target form: {piece_type}",
        type_guidance,
        "Length target: 500-1500 words unless shorter is artistically stronger.",
        "Output markdown only.",
        "First line must be a level-1 heading with the title: # <title>",
        "Do not include meta commentary about your writing process.",
    ]
    if directive:
        parts.extend(["", f"User directive: {directive}"])
    if creative_soul:
        parts.extend(["", "Creative voice rules:", creative_soul])
    if recent:
        parts.extend(
            [
                "",
                "Recent works to avoid repetition and support long-term evolution:",
                recent,
            ]
        )
    return "\n".join(parts)


def build_story_start_prompt(directive: str, creative_soul: str) -> str:
    schema = {
        "title": "string",
        "genre": "string",
        "premise": "string",
        "characters": [{"name": "string", "role": "string", "notes": "string"}],
        "themes": ["string"],
        "outline": [{"chapter": 1, "goal": "string", "conflict": "string", "turn": "string"}],
    }
    parts = [
        "Start a new long-form story project based on the directive.",
        "Return exactly two sections:",
        "1) A fenced JSON block with story planning state.",
        "2) The full text of Chapter 1 in markdown.",
        "The JSON block must follow this schema:",
        "```json",
        json.dumps(schema, ensure_ascii=False, indent=2),
        "```",
        "Chapter 1 should be 1200-2500 words and end with forward momentum.",
        "Chapter markdown must begin with '# Chapter 1: <title>'.",
        f"Directive: {directive}",
    ]
    if creative_soul:
        parts.extend(["", "Creative voice rules:", creative_soul])
    return "\n".join(parts)


def build_story_chapter_prompt(
    story: StoryState,
    creative_soul: str,
    last_chapter: str,
) -> str:
    state = json.dumps(story.to_dict(), ensure_ascii=False, indent=2)
    parts = [
        "Continue this long-form story with the next chapter.",
        "Keep continuity with existing details, tone, and unresolved threads.",
        "Output markdown only.",
        f"Write Chapter {story.chapters_completed + 1} with 1200-2500 words.",
        f"Start with '# Chapter {story.chapters_completed + 1}: <title>'.",
        "",
        "Current story state:",
        "```json",
        state,
        "```",
    ]
    if story.chapter_summaries:
        parts.append("\nPrevious chapter summaries:")
        parts.extend(
            f"- Chapter {i + 1}: {summary}" for i, summary in enumerate(story.chapter_summaries)
        )
    if last_chapter:
        parts.extend(["", "Previous chapter (full text):", last_chapter])
    if creative_soul:
        parts.extend(["", "Creative voice rules:", creative_soul])
    return "\n".join(parts)

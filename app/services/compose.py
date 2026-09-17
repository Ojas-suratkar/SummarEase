"""
Draft/Compose mode -- the opposite direction from everything else in this
app. Every other feature takes something that exists and extracts or
analyzes it; this takes a summary and *generates something new* from it:
an email, a social post, a follow-up message, a blog intro -- content you
didn't have before, meant to go back out into the world. Every
regeneration is saved as a new version rather than overwriting the last
one, and a version can be diffed against the one before it (word-level,
via Python's own `difflib` -- no new dependency), so refining a draft
across several tries is visible, not just a black box that spits out a
different answer each time you ask.
"""
from __future__ import annotations

import difflib
import html

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient
from ..extensions import db
from ..models import Draft

DRAFT_TYPES = {
    "email": "a professional email",
    "social_post": "a short, engaging social media post (under 280 characters)",
    "follow_up": "a polite follow-up message referencing this and asking a clarifying question",
    "blog_intro": "an attention-grabbing opening paragraph for a blog post",
}


class ComposeError(RuntimeError):
    pass


def generate_draft(source_text: str, draft_type: str, instructions: str = "") -> str:
    source_text = (source_text or "").strip()
    if not source_text:
        raise ComposeError("Nothing to draft from.")
    kind_description = DRAFT_TYPES.get(draft_type, DRAFT_TYPES["email"])

    prompt = (
        f"Based on the source material below, write {kind_description}. "
        "Write only the draft itself -- no preamble, no 'Here's a draft', "
        "no explanation."
        + (f" Additional instructions: {instructions.strip()}" if instructions.strip() else "")
        + f"\n\nSOURCE:\n{source_text[:6000]}"
    )
    try:
        response = generate_content_resilient(prompt)
        return (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise ComposeError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise ComposeError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise ComposeError(f"Draft generation failed: {exc}") from exc


def save_version(user_id: int, entry_id: int | None, draft_type: str, instructions: str, content: str) -> dict:
    last = (
        Draft.query.filter_by(user_id=user_id, entry_id=entry_id, draft_type=draft_type)
        .order_by(Draft.version.desc())
        .first()
    )
    version = (last.version if last else 0) + 1
    draft = Draft(
        user_id=user_id,
        entry_id=entry_id,
        draft_type=draft_type,
        instructions=instructions,
        content=content,
        version=version,
    )
    db.session.add(draft)
    db.session.commit()
    return {"id": draft.id, "version": version}


def list_versions(user_id: int, entry_id: int | None, draft_type: str) -> list[dict]:
    drafts = (
        Draft.query.filter_by(user_id=user_id, entry_id=entry_id, draft_type=draft_type)
        .order_by(Draft.version.desc())
        .all()
    )
    return [
        {
            "id": d.id,
            "entry_id": d.entry_id,
            "draft_type": d.draft_type,
            "instructions": d.instructions,
            "content": d.content,
            "version": d.version,
            "created_at": d.created_at.isoformat(),
        }
        for d in drafts
    ]


def word_diff_html(old_text: str, new_text: str) -> str:
    """A word-level inline diff (not a line diff -- these are short
    drafts, so word granularity reads much more clearly), rendered as
    plain HTML with inserted text in a green span and removed text in a
    red strikethrough span. difflib.SequenceMatcher does the actual
    comparison; this just walks its opcodes."""
    old_words = [html.escape(w) for w in old_text.split()]
    new_words = [html.escape(w) for w in new_text.split()]
    matcher = difflib.SequenceMatcher(a=old_words, b=new_words)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(new_words[j1:j2]))
        elif tag == "insert":
            parts.append(f'<span class="se-diff-add">{" ".join(new_words[j1:j2])}</span>')
        elif tag == "delete":
            parts.append(f'<span class="se-diff-del">{" ".join(old_words[i1:i2])}</span>')
        elif tag == "replace":
            parts.append(f'<span class="se-diff-del">{" ".join(old_words[i1:i2])}</span>')
            parts.append(f'<span class="se-diff-add">{" ".join(new_words[j1:j2])}</span>')
    return " ".join(parts)

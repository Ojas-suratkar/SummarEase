"""
Weekly digest -- a scheduled aggregation/reporting feature, not an
analysis-of-one-thing feature. Nothing else in this app looks back over
*everything that happened* in a time window and reports on it; every
other feature is scoped to a single source you hand it right now. This
pulls together what you've summarized and what your watchlist caught
changing over the last 7 days into one short "week in review" -- the
same shape as a real product's weekly digest email, generated from your
own data rather than written by hand.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from . import history_store, watchlist
from .cache import cached_call
from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient

_DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative": {
            "type": "string",
            "description": "A short (4-7 sentence), warm, specific summary of the week's reading and monitored changes -- not generic filler.",
        },
        "highlights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-6 short bullet-style highlights of the single most notable items.",
        },
    },
    "required": ["narrative", "highlights"],
}


class DigestError(RuntimeError):
    pass


def _week_key(now: datetime) -> str:
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _recent_entries(user_id: int, days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = history_store.list_recent(user_id, limit=500)
    return [e for e in entries if e.get("created_at", "") >= cutoff.isoformat()]


def _recent_digest_items(user_id: int, days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items = watchlist.list_digest(user_id, limit=200)
    return [i for i in items if i.get("created_at", "") >= cutoff.isoformat()]


def build_weekly_digest(user_id: int, days: int = 7) -> dict:
    """Returns the current week's digest, generating (and caching) it if
    this is the first request this week. Never errors outright: if
    there's nothing to report, or Gemini is unavailable, it returns a
    plain data-only digest instead of a narrative."""
    now = datetime.now(timezone.utc)
    entries = _recent_entries(user_id, days)
    watch_items = _recent_digest_items(user_id, days)

    if not entries and not watch_items:
        return {
            "narrative": "",
            "highlights": [],
            "entries": [],
            "watch_items": [],
            "period_days": days,
            "has_content": False,
        }

    entry_lines = "\n".join(
        f"- ({e['source_type']}) {e.get('source_ref', '')}: {e['summary'][:200]}" for e in entries
    )
    watch_lines = "\n".join(f"- {i['watch_label']}: {i['headline']}" for i in watch_items)
    prompt = (
        f"Here is everything summarized and everything a watchlist caught "
        f"changing over the last {days} days. Write a short, specific "
        "'week in review' -- reference actual topics, not generic phrases "
        "like 'a variety of topics'. Then list the 3-6 most notable "
        "individual items as highlights.\n\n"
        f"SUMMARIZED ({len(entries)} items):\n{entry_lines or '(none)'}\n\n"
        f"WATCHLIST CHANGES ({len(watch_items)} items):\n{watch_lines or '(none)'}"
    )

    def _compute() -> str:
        from google.genai import types

        response = generate_content_resilient(
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=_DIGEST_SCHEMA
            ),
        )
        return response.text

    narrative, highlights = "", []
    try:
        raw, _hit = cached_call(
            "weekly_digest",
            [str(user_id), _week_key(now), str(len(entries)), str(len(watch_items))],
            _compute,
            ttl_seconds=60 * 60 * 24,
        )
        data = json.loads(raw)
        narrative = data.get("narrative", "")
        highlights = data.get("highlights", [])
    except (GeminiNotConfiguredError, GeminiUnavailableError, json.JSONDecodeError, TypeError):
        # No narrative, but the raw data below is still a perfectly
        # useful digest on its own -- this never turns into a hard error.
        pass

    return {
        "narrative": narrative,
        "highlights": highlights,
        "entries": entries,
        "watch_items": watch_items,
        "period_days": days,
        "has_content": True,
    }

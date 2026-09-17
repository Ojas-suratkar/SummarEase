"""
Autonomous watchlist.

This is the one feature in this app that runs *without a request*. Every
other feature -- summarize, ask, compare -- only does something because a
browser tab sent it a request. A watchlist entry keeps working while no
one is looking at it: a background thread wakes on its own schedule,
re-fetches a page the user asked to be watched, and -- only when the page
has actually changed -- has Gemini write a short note on what's new and
drops it into a digest feed. That's the specific capability a chat
interface fundamentally cannot offer: ChatGPT/Claude can summarize a page
you paste in right now, but neither can notice, on their own, that a page
you asked about last week has changed since.

Design notes
------------
- Storage is SQLAlchemy (models.py's `Watch`/`DigestItem`), scoped by
  `user_id` on every query -- but the background scheduler itself is one
  process serving every account, so `_due_watches()` deliberately checks
  across all users at once; each individual `check_watch` call then reads
  the watch's own `user_id` to file its digest item under the right
  account.
- "Changed" is judged by content, not by HTTP headers/ETags (those are
  frequently missing or lie) -- the freshly extracted article text is
  hashed (SHA-256) and compared to the hash from the last successful
  check. A changed hash means "something worth a human look," not
  necessarily a large change -- the digest note itself, written by
  Gemini from a diff of old vs. new text, is what tells you whether it
  mattered.
- The background scheduler (`start_scheduler` in this module) is a
  single daemon thread with its own sleep loop, not a new dependency
  like APScheduler or Celery -- this app already has exactly this
  pattern for background jobs (services/jobs.py), just on a timer
  instead of triggered per-request. `create_app()` starts it once,
  inside a real Flask app context (see app_context.py) since it needs
  database access.
- Checking a watch never raises -- like history_store.save_entry, a
  failed check (page down, extraction failure, API hiccup) is recorded
  as a "last_error" on the watch and skipped, and the scheduler moves on
  to the next one. An autonomous background process that can crash
  itself out on one bad URL isn't autonomous.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from datetime import datetime, timezone

from . import app_context
from .article_extractor import ArticleExtractionError, extract_article_text
from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient
from ..extensions import db
from ..models import DigestItem, Watch

logger = logging.getLogger(__name__)

# How often the background thread wakes up to see which watches are due.
# Individual watches can have a longer interval than this -- this is just
# the scheduler's own resolution.
_SCHEDULER_TICK_SECONDS = 30

_scheduler_started = False
_scheduler_lock = threading.Lock()


class WatchlistError(RuntimeError):
    pass


def _watch_dict(w: Watch) -> dict:
    return {
        "id": w.id,
        "user_id": w.user_id,
        "url": w.url,
        "label": w.label,
        "interval_minutes": w.interval_minutes,
        "active": w.active,
        "last_checked_at": w.last_checked_at.isoformat() if w.last_checked_at else None,
        "last_content_hash": w.last_content_hash,
        "last_snapshot_text": w.last_snapshot_text,
        "last_error": w.last_error,
        "check_count": w.check_count,
        "created_at": w.created_at.isoformat(),
    }


def add_watch(user_id: int, url: str, label: str = "", interval_minutes: int = 60) -> int:
    url = (url or "").strip()
    if not url:
        raise WatchlistError("Please provide a URL to watch.")
    interval_minutes = max(1, int(interval_minutes or 60))
    watch = Watch(user_id=user_id, url=url, label=label.strip() or url, interval_minutes=interval_minutes)
    db.session.add(watch)
    db.session.commit()
    return watch.id


def remove_watch(watch_id: int, user_id: int) -> None:
    DigestItem.query.filter(
        DigestItem.watch_id == watch_id, DigestItem.user_id == user_id
    ).delete()
    Watch.query.filter_by(id=watch_id, user_id=user_id).delete()
    db.session.commit()


def toggle_watch(watch_id: int, user_id: int, active: bool) -> None:
    Watch.query.filter_by(id=watch_id, user_id=user_id).update({"active": active})
    db.session.commit()


def list_watches(user_id: int) -> list[dict]:
    watches = Watch.query.filter_by(user_id=user_id).order_by(Watch.id.desc()).all()
    return [_watch_dict(w) for w in watches]


def get_watch(watch_id: int, user_id: int) -> dict | None:
    watch = Watch.query.filter_by(id=watch_id, user_id=user_id).first()
    return _watch_dict(watch) if watch else None


def list_digest(user_id: int, limit: int = 50, unread_only: bool = False) -> list[dict]:
    query = DigestItem.query.filter_by(user_id=user_id)
    if unread_only:
        query = query.filter_by(is_read=False)
    items = query.order_by(DigestItem.id.desc()).limit(limit).all()
    return [
        {
            "id": i.id,
            "watch_id": i.watch_id,
            "headline": i.headline,
            "change_summary": i.change_summary,
            "is_read": i.is_read,
            "created_at": i.created_at.isoformat(),
            "watch_label": i.watch.label if i.watch else "",
            "watch_url": i.watch.url if i.watch else "",
        }
        for i in items
    ]


def health_stats(user_id: int) -> dict:
    """Aggregate watch health for the ops dashboard (routes.py's
    /dashboard) -- how many watches are active, and how many are
    currently failing (last check ended in an error)."""
    watches = list_watches(user_id)
    return {
        "total": len(watches),
        "active": sum(1 for w in watches if w["active"]),
        "failing": sum(1 for w in watches if w["last_error"]),
    }


def unread_count(user_id: int) -> int:
    return DigestItem.query.filter_by(user_id=user_id, is_read=False).count()


def mark_all_read(user_id: int) -> None:
    DigestItem.query.filter_by(user_id=user_id, is_read=False).update({"is_read": True})
    db.session.commit()


def _summarize_change(url: str, label: str, old_text: str, new_text: str) -> tuple[str, str]:
    """Ask Gemini to describe, in a sentence and a short paragraph, what
    actually changed between two snapshots of the same page. Returns
    (headline, change_summary)."""
    prompt = (
        f"You are monitoring the web page \"{label}\" ({url}) for a reader "
        "who doesn't have time to re-read the whole thing. Below are two "
        "snapshots of its main text content, OLD and NEW. Compare them and "
        "report only what's actually new or changed -- do not re-summarize "
        "content that's present in both.\n\n"
        "Respond in exactly two parts, separated by a line containing only "
        "'---':\n"
        "1. A single-sentence headline (under 15 words) describing the "
        "change.\n"
        "2. A short paragraph (2-4 sentences) explaining what's new, "
        "changed, or removed. If the change looks purely cosmetic "
        "(formatting, ads, unrelated boilerplate) say so plainly instead "
        "of inventing substance.\n\n"
        f"OLD:\n{old_text[:4000]}\n\nNEW:\n{new_text[:4000]}"
    )
    response = generate_content_resilient(prompt)
    raw = (response.text or "").strip()
    if "---" in raw:
        headline, _, body = raw.partition("---")
    else:
        headline, _, body = raw.partition("\n")
    headline = headline.strip().lstrip("#").strip() or "Page updated"
    body = body.strip() or raw
    return headline[:200], body[:2000]


def check_watch(watch: dict, report=lambda msg: None) -> dict:
    """Re-fetch one watch's URL, compare it to its last snapshot, and --
    only if the content actually changed -- write a digest entry. Always
    returns a status dict and never raises; failures are recorded on the
    watch itself so one bad URL never takes down the scheduler."""
    watch_id = watch["id"]
    user_id = watch["user_id"]
    now = datetime.now(timezone.utc)
    try:
        report(f"Checking {watch['label']}...")
        new_text = extract_article_text(watch["url"])
        new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()

        Watch.query.filter_by(id=watch_id).update(
            {"last_checked_at": now, "check_count": Watch.check_count + 1, "last_error": None}
        )
        db.session.commit()

        old_hash = watch.get("last_content_hash")
        old_text = watch.get("last_snapshot_text") or ""

        if old_hash == new_hash:
            return {"changed": False, "watch_id": watch_id}

        changed_meaningfully = old_hash is not None
        headline, change_summary = "First check -- now being tracked.", (
            "This is the first successful check of this watch. Future "
            "checks will report what changes."
        )
        if changed_meaningfully:
            report("Content changed -- asking Gemini what's new...")
            try:
                headline, change_summary = _summarize_change(
                    watch["url"], watch["label"], old_text, new_text
                )
            except GeminiNotConfiguredError:
                headline, change_summary = (
                    "Page content changed",
                    "The page's content changed since the last check, but a "
                    "Gemini API key isn't configured so a summary of the "
                    "change couldn't be generated.",
                )
            except Exception as exc:  # noqa: BLE001
                headline, change_summary = (
                    "Page content changed",
                    f"The page's content changed since the last check. "
                    f"(Couldn't generate a change summary: {exc})",
                )

        Watch.query.filter_by(id=watch_id).update(
            {"last_content_hash": new_hash, "last_snapshot_text": new_text[:20000]}
        )
        if changed_meaningfully:
            db.session.add(
                DigestItem(
                    user_id=user_id, watch_id=watch_id, headline=headline, change_summary=change_summary
                )
            )
        db.session.commit()

        return {"changed": changed_meaningfully, "watch_id": watch_id, "headline": headline}

    except ArticleExtractionError as exc:
        Watch.query.filter_by(id=watch_id).update(
            {"last_checked_at": now, "check_count": Watch.check_count + 1, "last_error": str(exc)}
        )
        db.session.commit()
        return {"changed": False, "watch_id": watch_id, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- a background loop must never die on one watch
        logger.warning("Watch %s check failed: %s", watch_id, exc)
        Watch.query.filter_by(id=watch_id).update(
            {"last_checked_at": now, "check_count": Watch.check_count + 1, "last_error": str(exc)}
        )
        db.session.commit()
        return {"changed": False, "watch_id": watch_id, "error": str(exc)}


def check_now(watch_id: int, user_id: int) -> dict:
    """Force an immediate check of one watch, bypassing its interval --
    used by the 'Check now' button so a demo doesn't have to wait."""
    watch = get_watch(watch_id, user_id)
    if watch is None:
        raise WatchlistError("Watch not found.")
    return check_watch(watch)


def _due_watches() -> list[dict]:
    """Every active watch, across every account, whose interval has
    elapsed -- the scheduler serves all users from one background
    thread, unlike everything else in this module which is per-user."""
    watches = Watch.query.filter_by(active=True).all()
    due = []
    now = datetime.now(timezone.utc)
    for w in watches:
        if w.last_checked_at is None:
            due.append(_watch_dict(w))
            continue
        # SQLite hands datetimes back naive even though they were stored
        # timezone-aware -- normalize before subtracting, or this raises
        # "can't compare offset-naive and offset-aware datetimes" as soon
        # as a watch has been checked once (Postgres doesn't have this
        # issue, but the app must run correctly on both).
        last_checked = w.last_checked_at
        if last_checked.tzinfo is None:
            last_checked = last_checked.replace(tzinfo=timezone.utc)
        elapsed_minutes = (now - last_checked).total_seconds() / 60
        if elapsed_minutes >= w.interval_minutes:
            due.append(_watch_dict(w))
    return due


def check_all_due() -> int:
    """Check every active watch (any account) whose interval has
    elapsed. Returns how many were checked. Safe to call repeatedly --
    each individual check is isolated and failures don't stop the rest."""
    due = _due_watches()
    for watch in due:
        check_watch(watch)
    return len(due)


def _scheduler_loop() -> None:
    while True:
        try:
            app_context.run(check_all_due)
        except Exception as exc:  # noqa: BLE001 -- the loop itself must never die
            logger.warning("Watchlist scheduler tick failed: %s", exc)
        time.sleep(_SCHEDULER_TICK_SECONDS)


def start_scheduler() -> None:
    """Start the background polling thread exactly once per process.
    Safe to call from create_app() even under Flask's debug reloader
    (which imports the app twice) -- the module-level flag ensures only
    one thread ever runs per interpreter."""
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True
    thread = threading.Thread(target=_scheduler_loop, daemon=True, name="watchlist-scheduler")
    thread.start()

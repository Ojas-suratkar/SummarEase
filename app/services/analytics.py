"""
Personal analytics -- computed entirely from your own stored rows, with
zero Gemini calls. This is the app reporting back on how *you've* used
it: how much you've read, what kinds of sources, how well flashcard
review is actually sticking, and how active your watchlist is. Plain
aggregation over SQLAlchemy queries, not an AI feature.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from ..models import Annotation, DigestItem, Draft, Flashcard, HistoryEntry, Watch


def _as_aware_utc(dt: datetime | None) -> datetime | None:
    """SQLite (unlike Postgres) hands datetimes back naive even though
    they were stored timezone-aware -- normalize so comparisons below
    never raise "can't compare offset-naive and offset-aware datetimes"
    regardless of which database this is running against."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def get_analytics(user_id: int) -> dict:
    entries = HistoryEntry.query.filter_by(user_id=user_id).order_by(HistoryEntry.created_at.asc()).all()
    entry_count = len(entries)

    by_type = Counter(e.source_type for e in entries)

    now = datetime.now(timezone.utc)
    cutoff_30 = now - timedelta(days=30)
    daily_counts: Counter = Counter()
    for e in entries:
        created = _as_aware_utc(e.created_at)
        if created and created >= cutoff_30:
            daily_counts[created.date().isoformat()] += 1

    total_words = sum(len((e.source_text or "").split()) for e in entries)
    avg_summary_words = (
        round(sum(len((e.summary or "").split()) for e in entries) / entry_count, 1)
        if entry_count
        else 0
    )

    flashcards = Flashcard.query.filter_by(user_id=user_id).all()
    reviewed = [c for c in flashcards if c.last_reviewed_at is not None]
    retention_rate = (
        round(sum(1 for c in reviewed if c.repetitions > 0) / len(reviewed), 3) if reviewed else None
    )

    watches = Watch.query.filter_by(user_id=user_id).all()

    dates_with_activity = {_as_aware_utc(e.created_at).date() for e in entries if e.created_at}
    streak = 0
    day = now.date()
    while day in dates_with_activity:
        streak += 1
        day -= timedelta(days=1)

    return {
        "entry_count": entry_count,
        "by_source_type": dict(by_type),
        "daily_counts_30d": dict(sorted(daily_counts.items())),
        "total_words_processed": total_words,
        "avg_summary_words": avg_summary_words,
        "flashcard_count": len(flashcards),
        "flashcard_reviewed_count": len(reviewed),
        "flashcard_retention_rate": retention_rate,
        "watch_count": len(watches),
        "active_watch_count": sum(1 for w in watches if w.active),
        "digest_item_count": DigestItem.query.filter_by(user_id=user_id).count(),
        "annotation_count": Annotation.query.filter_by(user_id=user_id).count(),
        "draft_count": Draft.query.filter_by(user_id=user_id).count(),
        "current_streak_days": streak,
    }

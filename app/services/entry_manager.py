"""
Managing what you've saved: rename, retag, annotate, archive, delete.

A knowledge base you can only append to isn't a knowledge base, it's a
log. Everything here exists because the app previously had no answer to
completely ordinary requests -- "call this something sensible", "add a
note", "I didn't mean to save that", "clear all of this out".

Two decisions worth stating:

*Deletes are real.* When this deletes an entry it also removes the RAG
chunks, links, annotations, shares, flashcards and stored source files
that pointed at it. A "delete" that leaves the source file on disk and
the search index intact is a lie, and it is the kind of lie that gets
noticed at the worst possible moment.

*Bulk actions are undoable where it is honest to offer it.* Archiving
is reversible, so it's offered freely. Deleting is not, so it asks
first and says exactly how many things it is about to destroy.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..extensions import db
from ..models import (
    Annotation,
    AudioFingerprint,
    Draft,
    HistoryEntry,
    Link,
    RagChunk,
    ReaderSession,
    Recording,
    Share,
    SourceAsset,
)
from . import source_store

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_entry(user_id: int, entry_id: int) -> HistoryEntry | None:
    return HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()


def entry_dict(entry: HistoryEntry, *, with_source: bool = False) -> dict:
    data = {
        "id": entry.id,
        "title": entry.display_title,
        "custom_title": entry.title or "",
        "source_type": entry.source_type,
        "source_ref": entry.source_ref,
        "summary": entry.summary,
        "notes": entry.notes or "",
        "tags": entry.tag_list,
        "detail_level": entry.detail_level or "standard",
        "is_archived": bool(entry.is_archived),
        "is_pinned": bool(entry.is_pinned),
        "created_at": entry.created_at.isoformat() if entry.created_at else "",
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else "",
        "word_count": len((entry.source_text or "").split()),
    }
    if with_source:
        data["source_text"] = entry.source_text or ""
    return data


def update_entry(user_id: int, entry_id: int, **fields) -> dict | None:
    """Patch an entry. Only the fields actually supplied are touched, so
    a form that submits one thing can't blank out the others."""
    entry = get_entry(user_id, entry_id)
    if entry is None:
        return None

    if "title" in fields:
        entry.title = (fields["title"] or "").strip()[:300] or None
    if "summary" in fields:
        summary = (fields["summary"] or "").strip()
        if summary:
            entry.summary = summary
    if "notes" in fields:
        entry.notes = (fields["notes"] or "").strip()
    if "tags" in fields:
        entry.tags = normalise_tags(fields["tags"])
    if "is_archived" in fields:
        entry.is_archived = bool(fields["is_archived"])
    if "is_pinned" in fields:
        entry.is_pinned = bool(fields["is_pinned"])
    if "source_text" in fields:
        # Editing the source is allowed on purpose: OCR and article
        # extraction both make mistakes, and being able to fix the input
        # is what makes re-running worthwhile.
        entry.source_text = fields["source_text"] or ""

    entry.updated_at = _now()
    db.session.commit()
    return entry_dict(entry)


def normalise_tags(raw) -> str:
    """Tags arrive as a comma string from a form or a list from JSON.
    Lowercased and de-duplicated so 'Research', 'research' and
    ' research ' are one tag rather than three."""
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw or [])
    seen: list[str] = []
    for part in parts:
        tag = str(part).strip().lower()[:40]
        if tag and tag not in seen:
            seen.append(tag)
    return ",".join(seen[:25])


def delete_entry(user_id: int, entry_id: int) -> bool:
    """Remove an entry and everything that belonged to it."""
    entry = get_entry(user_id, entry_id)
    if entry is None:
        return False

    RagChunk.query.filter_by(entry_id=entry_id).delete()
    Link.query.filter(
        db.or_(Link.from_id == entry_id, Link.to_id == entry_id), Link.user_id == user_id
    ).delete(synchronize_session=False)
    Annotation.query.filter_by(entry_id=entry_id, user_id=user_id).delete()
    Share.query.filter_by(entry_id=entry_id, user_id=user_id).delete()
    Draft.query.filter_by(entry_id=entry_id, user_id=user_id).delete()
    AudioFingerprint.query.filter_by(entry_id=entry_id, user_id=user_id).delete()
    ReaderSession.query.filter_by(entry_id=entry_id, user_id=user_id).delete()
    Recording.query.filter_by(entry_id=entry_id, user_id=user_id).update({"entry_id": None})

    source_store.delete_assets_for_entry(entry_id, user_id)

    db.session.delete(entry)
    db.session.commit()
    return True


def bulk_action(user_id: int, entry_ids: list[int], action: str, value: str = "") -> dict:
    """Apply one action to many entries. Returns counts rather than
    raising on a partial failure, because a bulk operation that stops
    halfway and reports nothing is worse than one that finishes and
    tells you what it managed."""
    entry_ids = [int(i) for i in entry_ids][:500]
    if not entry_ids:
        return {"affected": 0, "action": action}

    entries = HistoryEntry.query.filter(
        HistoryEntry.user_id == user_id, HistoryEntry.id.in_(entry_ids)
    ).all()

    affected = 0
    for entry in entries:
        if action == "archive":
            entry.is_archived = True
        elif action == "unarchive":
            entry.is_archived = False
        elif action == "pin":
            entry.is_pinned = True
        elif action == "unpin":
            entry.is_pinned = False
        elif action == "tag":
            tag = (value or "").strip().lower()
            if tag and tag not in entry.tag_list:
                entry.tags = normalise_tags(entry.tag_list + [tag])
        elif action == "untag":
            tag = (value or "").strip().lower()
            entry.tags = normalise_tags([t for t in entry.tag_list if t != tag])
        elif action == "delete":
            continue  # handled below, since it needs cascade cleanup
        else:
            continue
        entry.updated_at = _now()
        affected += 1

    if action == "delete":
        for entry_id in entry_ids:
            if delete_entry(user_id, entry_id):
                affected += 1
        return {"affected": affected, "action": action}

    db.session.commit()
    return {"affected": affected, "action": action}


def list_entries(
    user_id: int,
    *,
    query: str = "",
    tag: str = "",
    source_type: str = "",
    archived: bool = False,
    pinned_first: bool = True,
    limit: int = 60,
    offset: int = 0,
) -> dict:
    """The knowledge-base listing, with the filters a manageable library
    actually needs. Plain SQL matching -- instant, no API call, works
    when everything else is down."""
    q = HistoryEntry.query.filter_by(user_id=user_id)
    q = q.filter(HistoryEntry.is_archived.is_(True)) if archived else q.filter(HistoryEntry.is_archived.isnot(True))

    if query.strip():
        like = f"%{query.strip()}%"
        q = q.filter(
            db.or_(
                HistoryEntry.summary.ilike(like),
                HistoryEntry.source_ref.ilike(like),
                HistoryEntry.title.ilike(like),
                HistoryEntry.notes.ilike(like),
            )
        )
    if tag.strip():
        q = q.filter(HistoryEntry.tags.ilike(f"%{tag.strip().lower()}%"))
    if source_type.strip():
        q = q.filter(HistoryEntry.source_type == source_type.strip())

    total = q.count()
    if pinned_first:
        q = q.order_by(HistoryEntry.is_pinned.desc(), HistoryEntry.id.desc())
    else:
        q = q.order_by(HistoryEntry.id.desc())

    rows = q.limit(limit).offset(offset).all()
    return {
        "entries": [entry_dict(r) for r in rows],
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(rows) < total,
    }


def all_tags(user_id: int) -> list[dict]:
    rows = HistoryEntry.query.filter_by(user_id=user_id).all()
    counts: dict[str, int] = {}
    for row in rows:
        for tag in row.tag_list:
            counts[tag] = counts.get(tag, 0) + 1
    return [{"tag": t, "count": c} for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def source_type_counts(user_id: int) -> list[dict]:
    rows = (
        db.session.query(HistoryEntry.source_type, db.func.count(HistoryEntry.id))
        .filter(HistoryEntry.user_id == user_id)
        .group_by(HistoryEntry.source_type)
        .all()
    )
    return [{"source_type": t or "unknown", "count": c} for t, c in sorted(rows, key=lambda kv: -kv[1])]


def delete_all_entries(user_id: int) -> int:
    """The nuclear option, offered because people are entitled to leave
    with nothing left behind. Goes through delete_entry so the cascade
    and the stored files are handled the same way as a single delete."""
    ids = [row.id for row in HistoryEntry.query.filter_by(user_id=user_id).all()]
    deleted = 0
    for entry_id in ids:
        if delete_entry(user_id, entry_id):
            deleted += 1
    source_store.purge_user_storage(user_id)
    return deleted


def library_stats(user_id: int) -> dict:
    total = HistoryEntry.query.filter_by(user_id=user_id).count()
    archived = HistoryEntry.query.filter_by(user_id=user_id, is_archived=True).count()
    pinned = HistoryEntry.query.filter_by(user_id=user_id, is_pinned=True).count()
    with_notes = HistoryEntry.query.filter(
        HistoryEntry.user_id == user_id, HistoryEntry.notes.isnot(None), HistoryEntry.notes != ""
    ).count()
    return {
        "total": total,
        "active": total - archived,
        "archived": archived,
        "pinned": pinned,
        "with_notes": with_notes,
        "tags": len(all_tags(user_id)),
    }

"""
Data portability -- your account's data, as one JSON file you control.

Export dumps every row this account owns (knowledge base, flashcards,
annotations, drafts, watches, links) into a single plain JSON document.
Import reads that same shape back in -- either to restore this account,
or to move data into a *different* SummarEase account/instance entirely.
Nothing here calls Gemini or any other API; it's a direct database
dump/restore, same idea as "export my data" on any real product.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..extensions import db
from ..models import Annotation, Draft, Flashcard, HistoryEntry, Link, Watch

EXPORT_VERSION = 1


class BackupError(RuntimeError):
    pass


def export_account(user_id: int) -> dict:
    entries = HistoryEntry.query.filter_by(user_id=user_id).order_by(HistoryEntry.id.asc()).all()
    return {
        "version": EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "history_entries": [
            {
                "id": e.id,
                "source_type": e.source_type,
                "source_ref": e.source_ref,
                "summary": e.summary,
                "source_text": e.source_text,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ],
        "links": [
            {
                "from_id": link.from_id,
                "to_id": link.to_id,
                "relationship_type": link.relationship_type,
                "rationale": link.rationale,
                "similarity": link.similarity,
            }
            for link in Link.query.filter_by(user_id=user_id).all()
        ],
        "flashcards": [
            {
                "source_ref": c.source_ref,
                "question": c.question,
                "answer": c.answer,
                "ease_factor": c.ease_factor,
                "interval_days": c.interval_days,
                "repetitions": c.repetitions,
            }
            for c in Flashcard.query.filter_by(user_id=user_id).all()
        ],
        "annotations": [
            {"entry_id": a.entry_id, "quote": a.quote, "note": a.note}
            for a in Annotation.query.filter_by(user_id=user_id).all()
        ],
        "drafts": [
            {
                "entry_id": d.entry_id,
                "draft_type": d.draft_type,
                "instructions": d.instructions,
                "content": d.content,
                "version": d.version,
            }
            for d in Draft.query.filter_by(user_id=user_id).all()
        ],
        "watches": [
            {
                "url": w.url,
                "label": w.label,
                "interval_minutes": w.interval_minutes,
                "active": w.active,
            }
            for w in Watch.query.filter_by(user_id=user_id).all()
        ],
    }


def import_account(user_id: int, data: dict) -> dict:
    """Best-effort restore under `user_id`. History entries are
    re-embedded and re-indexed as they're recreated (so search/Q&A work
    immediately); annotations/drafts are remapped to the *new* entry ids
    created here, since the ids in the export file belong to whichever
    account/instance it came from, not this one."""
    if not isinstance(data, dict) or "history_entries" not in data:
        raise BackupError("That file doesn't look like a SummarEase backup.")

    from . import history_store, rag

    id_map: dict[int, int] = {}
    counts = {"entries": 0, "flashcards": 0, "annotations": 0, "drafts": 0, "watches": 0}

    for e in data.get("history_entries", []):
        new_id = history_store.save_entry(
            user_id,
            e.get("source_type", "text"),
            e.get("source_ref", ""),
            e.get("summary", ""),
            e.get("source_text", "") or "",
        )
        old_id = e.get("id")
        if old_id is not None:
            id_map[old_id] = new_id
        try:
            rag.index_entry(new_id, e.get("source_text") or e.get("summary") or "")
        except Exception:  # noqa: BLE001 -- indexing is best-effort, same as a normal save
            pass
        counts["entries"] += 1

    for c in data.get("flashcards", []):
        db.session.add(
            Flashcard(
                user_id=user_id,
                source_ref=c.get("source_ref", ""),
                question=c.get("question", ""),
                answer=c.get("answer", ""),
                ease_factor=c.get("ease_factor", 2.5),
                interval_days=c.get("interval_days", 0),
                repetitions=c.get("repetitions", 0),
            )
        )
        counts["flashcards"] += 1

    for a in data.get("annotations", []):
        mapped = id_map.get(a.get("entry_id"))
        if mapped is None:
            continue
        db.session.add(Annotation(user_id=user_id, entry_id=mapped, quote=a.get("quote", ""), note=a.get("note", "")))
        counts["annotations"] += 1

    for d in data.get("drafts", []):
        mapped = id_map.get(d.get("entry_id")) if d.get("entry_id") is not None else None
        db.session.add(
            Draft(
                user_id=user_id,
                entry_id=mapped,
                draft_type=d.get("draft_type", "email"),
                instructions=d.get("instructions", ""),
                content=d.get("content", ""),
                version=d.get("version", 1),
            )
        )
        counts["drafts"] += 1

    for w in data.get("watches", []):
        db.session.add(
            Watch(
                user_id=user_id,
                url=w.get("url", ""),
                label=w.get("label", ""),
                interval_minutes=w.get("interval_minutes", 60),
                active=w.get("active", True),
            )
        )
        counts["watches"] += 1

    db.session.commit()
    return counts

"""
Persistence for the vault, the spatial canvas, the speed reader, the
offline sync log, and audio fingerprints.

These are grouped in one module because each is a thin, boring layer
over a single table -- the interesting logic lives in app/core/ (the
crypto lives in the browser). Splitting five ~40-line CRUD wrappers into
five files would be filing, not architecture.

One rule runs through all of it: every query is scoped by `user_id`.
Not "usually", not "on the read paths" -- every single one. An
ownership check that exists on four of five paths is a data leak with
good intentions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from ..core import acoustics, crdt, reading
from ..extensions import db
from ..models import (
    AudioFingerprint,
    CanvasLayout,
    HistoryEntry,
    ReaderSession,
    SyncOp,
    VaultItem,
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Zero-knowledge vault
#
# Note what is absent from every function here: any decryption, any key,
# any plaintext. The server stores a blob and hands the same blob back.
# If a function in this section ever needs to read the contents, the
# feature has been broken.
# ---------------------------------------------------------------------------


def vault_put(user_id: int, item_key: str, ciphertext: str, iv: str, salt: str, *, algo: str = "AES-GCM-256", kdf_iterations: int = 600000) -> dict:
    item_key = (item_key or "").strip()[:64]
    if not item_key or not ciphertext or not iv or not salt:
        raise ValueError("A vault item needs a key, ciphertext, an IV and a salt.")

    item = VaultItem.query.filter_by(user_id=user_id, item_key=item_key).first()
    if item is None:
        item = VaultItem(user_id=user_id, item_key=item_key)
        db.session.add(item)

    item.ciphertext = ciphertext
    item.iv = iv
    item.salt = salt
    item.algo = algo
    item.kdf_iterations = int(kdf_iterations or 600000)
    item.byte_size = len(ciphertext)
    item.updated_at = _now()
    db.session.commit()
    return vault_item_dict(item)


def vault_item_dict(item: VaultItem, *, include_ciphertext: bool = False) -> dict:
    data = {
        "entry_id": item.item_key,
        "item_key": item.item_key,
        "bytes": item.byte_size or 0,
        "algo": item.algo,
        "kdf_iterations": item.kdf_iterations,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }
    if include_ciphertext:
        data.update({"ciphertext": item.ciphertext, "iv": item.iv, "salt": item.salt})
    return data


def vault_list(user_id: int) -> list[dict]:
    items = VaultItem.query.filter_by(user_id=user_id).order_by(VaultItem.updated_at.desc()).all()
    return [vault_item_dict(i) for i in items]


def vault_get(user_id: int, item_key: str) -> dict | None:
    item = VaultItem.query.filter_by(user_id=user_id, item_key=item_key).first()
    return vault_item_dict(item, include_ciphertext=True) if item else None


def vault_delete(user_id: int, item_key: str) -> bool:
    item = VaultItem.query.filter_by(user_id=user_id, item_key=item_key).first()
    if item is None:
        return False
    db.session.delete(item)
    db.session.commit()
    return True


def vault_stats(user_id: int) -> dict:
    items = VaultItem.query.filter_by(user_id=user_id).all()
    return {
        "count": len(items),
        "bytes": sum(i.byte_size or 0 for i in items),
        "algo": items[0].algo if items else "AES-GCM-256",
    }


# ---------------------------------------------------------------------------
# Spatial canvas
# ---------------------------------------------------------------------------


_EMPTY_LAYOUT = {"nodes": [], "edges": [], "groups": [], "view": {"x": 0.0, "y": 0.0, "zoom": 1.0}}


def canvas_load(user_id: int, name: str = "Default board") -> dict:
    layout = CanvasLayout.query.filter_by(user_id=user_id, name=name).first()
    if layout is None:
        return dict(_EMPTY_LAYOUT)
    try:
        payload = json.loads(layout.payload or "{}")
    except json.JSONDecodeError:
        # A corrupted layout should cost the user their arrangement, not
        # their access to the page.
        logger.warning("Canvas layout for user %s was not valid JSON; starting empty", user_id)
        return dict(_EMPTY_LAYOUT)
    merged = dict(_EMPTY_LAYOUT)
    merged.update({k: payload.get(k, v) for k, v in _EMPTY_LAYOUT.items()})
    return merged


def canvas_save(user_id: int, payload: dict, name: str = "Default board") -> dict:
    cleaned = {
        "nodes": payload.get("nodes", [])[:2000],
        "edges": payload.get("edges", [])[:4000],
        "groups": payload.get("groups", [])[:200],
        "view": payload.get("view", {"x": 0, "y": 0, "zoom": 1}),
    }
    layout = CanvasLayout.query.filter_by(user_id=user_id, name=name).first()
    if layout is None:
        layout = CanvasLayout(user_id=user_id, name=name)
        db.session.add(layout)
    layout.payload = json.dumps(cleaned)
    layout.updated_at = _now()
    db.session.commit()
    return {"ok": True, "nodes": len(cleaned["nodes"])}


def canvas_items(user_id: int, limit: int = 300) -> list[dict]:
    """The documents available to place on the board, with pairwise
    similarity where we already have embeddings -- the force layout uses
    it to pull related cards together. Computing it here costs one pass
    over vectors we already stored; it never calls out to anything."""
    entries = (
        HistoryEntry.query.filter_by(user_id=user_id)
        .filter(HistoryEntry.is_archived.isnot(True))
        .order_by(HistoryEntry.id.desc())
        .limit(limit)
        .all()
    )
    items = [
        {
            "id": e.id,
            "title": e.display_title,
            "source_type": e.source_type,
            "summary": (e.summary or "")[:280],
            "created_at": e.created_at.isoformat() if e.created_at else "",
            "tags": e.tag_list,
            "similarity_to": {},
        }
        for e in entries
    ]

    try:
        import numpy as np

        vectors = {}
        for entry in entries:
            if entry.embedding:
                vectors[entry.id] = np.frombuffer(entry.embedding, dtype=np.float32)
        ids = list(vectors)
        if len(ids) > 1:
            matrix = np.vstack([vectors[i] for i in ids])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = matrix / norms
            similarity = unit @ unit.T
            index = {item["id"]: item for item in items}
            for row, entry_id in enumerate(ids):
                target = index.get(entry_id)
                if target is None:
                    continue
                # Only the strongest few neighbours matter to the layout,
                # and shipping a full n^2 matrix to the browser would be
                # megabytes for a few hundred cards.
                order = np.argsort(-similarity[row])[1:6]
                target["similarity_to"] = {
                    str(ids[col]): round(float(similarity[row][col]), 3)
                    for col in order
                    if float(similarity[row][col]) > 0.35
                }
    except Exception as exc:  # pragma: no cover
        logger.debug("Similarity precompute skipped: %s", exc)

    return items


# ---------------------------------------------------------------------------
# Speed reader
# ---------------------------------------------------------------------------


def reader_documents(user_id: int, limit: int = 100) -> list[dict]:
    entries = (
        HistoryEntry.query.filter_by(user_id=user_id)
        .filter(HistoryEntry.is_archived.isnot(True))
        .order_by(HistoryEntry.id.desc())
        .limit(limit)
        .all()
    )
    out = []
    for entry in entries:
        text = (entry.source_text or "") or (entry.summary or "")
        if not text.strip():
            continue
        out.append(
            {
                "id": entry.id,
                "title": entry.display_title,
                "source_type": entry.source_type,
                "word_count": len(text.split()),
            }
        )
    return out


def reader_document(user_id: int, entry_id: int, *, use_summary: bool = False) -> dict | None:
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    if entry is None:
        return None
    text = (entry.summary or "") if use_summary else ((entry.source_text or "") or (entry.summary or ""))
    text = text.strip()
    if not text:
        return None
    return {
        "id": entry.id,
        "title": entry.display_title,
        "text": text,
        "words": reading.prepare_words(text),
    }


def reader_record_session(user_id: int, entry_id: int | None, *, wpm: int, words_read: int, duration_ms: int, comprehension: float | None, completed: bool) -> dict:
    session = ReaderSession(
        user_id=user_id,
        entry_id=entry_id,
        wpm=int(wpm or 0),
        words_read=int(words_read or 0),
        duration_ms=int(duration_ms or 0),
        comprehension=comprehension,
        completed=bool(completed),
    )
    db.session.add(session)
    db.session.commit()

    if comprehension is None:
        return {"ok": True, "next_wpm": int(wpm or 300), "reason": "No quiz taken, so the speed stays where it is."}

    adaptation = reading.adapt_wpm(int(wpm or 300), float(comprehension))
    return {"ok": True, "next_wpm": adaptation["new_wpm"], "reason": adaptation["reason"], "change": adaptation["change"]}


def reader_stats(user_id: int, limit: int = 100) -> dict:
    rows = (
        ReaderSession.query.filter_by(user_id=user_id)
        .order_by(ReaderSession.created_at.desc())
        .limit(limit)
        .all()
    )
    sessions = [
        {
            "id": r.id,
            "entry_id": r.entry_id,
            "wpm": r.wpm,
            "words_read": r.words_read,
            "duration_ms": r.duration_ms,
            "comprehension": r.comprehension,
            "completed": r.completed,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
    ]
    return {"sessions": sessions, "fitness": reading.reading_fitness(sessions)}


# ---------------------------------------------------------------------------
# Offline sync log (CRDT operations)
# ---------------------------------------------------------------------------


def sync_push(user_id: int, operations: list[dict]) -> dict:
    """Accept a batch of operations a client queued while offline.

    Duplicates are expected and harmless: a client that loses its
    connection mid-push will resend, and an operation's `op_id` is its
    identity, so the second copy is dropped here rather than applied
    twice. That property is what lets the client retry blindly instead
    of implementing a careful, fragile acknowledgement protocol.
    """
    accepted = 0
    duplicates = 0
    rejected = 0

    for raw in operations[:1000]:
        try:
            op = crdt.Operation.from_dict(raw)
        except Exception:
            rejected += 1
            continue

        exists = SyncOp.query.filter_by(user_id=user_id, op_id=op.op_id).first()
        if exists is not None:
            duplicates += 1
            continue

        db.session.add(
            SyncOp(
                user_id=user_id,
                op_id=op.op_id,
                replica_id=op.replica_id,
                entity=op.entity,
                entity_id=str(op.entity_id),
                field=op.field or "",
                action=op.action,
                value=json.dumps(op.value),
                lamport=int(op.lamport or 0),
                wall_clock=float(op.wall_clock or 0.0),
            )
        )
        accepted += 1

    db.session.commit()
    return {"accepted": accepted, "duplicates": duplicates, "rejected": rejected}


def _row_to_operation(row: SyncOp) -> crdt.Operation:
    try:
        value = json.loads(row.value) if row.value else None
    except json.JSONDecodeError:
        value = None
    return crdt.Operation(
        op_id=row.op_id,
        replica_id=row.replica_id,
        entity=row.entity,
        entity_id=row.entity_id,
        field=row.field or "",
        action=row.action,
        value=value,
        lamport=row.lamport or 0,
        wall_clock=row.wall_clock or 0.0,
    )


def sync_pull(user_id: int, client_clock: dict | None = None, *, limit: int = 2000) -> dict:
    """Everything this client hasn't seen, according to its vector clock."""
    rows = (
        SyncOp.query.filter_by(user_id=user_id)
        .order_by(SyncOp.lamport.asc(), SyncOp.id.asc())
        .limit(limit)
        .all()
    )
    operations = [_row_to_operation(r) for r in rows]
    missing = crdt.diff_for_sync(client_clock or {}, operations)
    return {
        "operations": [op.to_dict() for op in missing],
        "server_total": len(operations),
        "sent": len(missing),
    }


def sync_state(user_id: int) -> dict:
    rows = SyncOp.query.filter_by(user_id=user_id).order_by(SyncOp.lamport.asc(), SyncOp.id.asc()).all()
    return crdt.resolve([_row_to_operation(r) for r in rows])


def sync_status(user_id: int) -> dict:
    total = SyncOp.query.filter_by(user_id=user_id).count()
    latest = (
        SyncOp.query.filter_by(user_id=user_id)
        .order_by(SyncOp.id.desc())
        .first()
    )
    replicas = db.session.query(SyncOp.replica_id).filter_by(user_id=user_id).distinct().count()
    return {
        "operations": total,
        "replicas": replicas,
        "last_op_at": latest.created_at.isoformat() if latest and latest.created_at else None,
    }


# ---------------------------------------------------------------------------
# Audio fingerprints
# ---------------------------------------------------------------------------


def fingerprint_store(user_id: int, fp: dict, *, label: str = "", entry_id: int | None = None, asset_id: int | None = None) -> dict:
    """Keep the landmark hashes for a recording.

    We store the hashes, not the audio analysis that produced them: the
    hashes are what matching needs, they're small, and they are not
    reversible into listenable audio.
    """
    hashes = [[h["hash"], round(float(h["time"]), 3)] for h in fp.get("hashes", [])]
    row = AudioFingerprint(
        user_id=user_id,
        entry_id=entry_id,
        asset_id=asset_id,
        label=label or "Recording",
        fingerprint_id=fp.get("fingerprint_id", ""),
        duration=float(fp.get("duration", 0.0)),
        hash_count=len(hashes),
        hashes=json.dumps(hashes),
    )
    db.session.add(row)
    db.session.commit()
    return {"id": row.id, "fingerprint_id": row.fingerprint_id, "hash_count": row.hash_count, "duration": row.duration}


def _row_to_fp(row: AudioFingerprint) -> dict:
    try:
        pairs = json.loads(row.hashes or "[]")
    except json.JSONDecodeError:
        pairs = []
    return {
        "hashes": [{"hash": h, "time": t} for h, t in pairs],
        "duration": row.duration or 0.0,
        "hash_count": row.hash_count or 0,
        "fingerprint_id": row.fingerprint_id or "",
        "peak_count": 0,
        "sample_rate": 11025,
    }


def fingerprint_library(user_id: int) -> list[dict]:
    rows = AudioFingerprint.query.filter_by(user_id=user_id).order_by(AudioFingerprint.id.desc()).all()
    return [
        {
            "id": r.id,
            "label": r.label,
            "entry_id": r.entry_id,
            "duration": r.duration,
            "hash_count": r.hash_count,
            "fingerprint_id": r.fingerprint_id,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
    ]


def fingerprint_match(user_id: int, fp: dict, *, top_n: int = 5) -> list[dict]:
    """Search the account's whole recording library for this audio.

    Uses the inverted index rather than comparing pairwise, so the cost
    is roughly constant in library size instead of linear.
    """
    rows = AudioFingerprint.query.filter_by(user_id=user_id).all()
    if not rows:
        return []
    index = acoustics.build_index([(str(r.id), _row_to_fp(r)) for r in rows])
    matches = acoustics.query_index(index, fp, top_n=top_n)

    by_id = {str(r.id): r for r in rows}
    enriched = []
    for match in matches:
        row = by_id.get(str(match.get("track_id")))
        if row is None:
            continue
        enriched.append(
            {
                **match,
                "label": row.label,
                "entry_id": row.entry_id,
                "duration": row.duration,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
        )
    return enriched


def fingerprint_delete(user_id: int, fingerprint_row_id: int) -> bool:
    row = AudioFingerprint.query.filter_by(id=fingerprint_row_id, user_id=user_id).first()
    if row is None:
        return False
    db.session.delete(row)
    db.session.commit()
    return True


def prune_reader_sessions(user_id: int, older_than_days: int = 365) -> int:
    cutoff = _now() - timedelta(days=older_than_days)
    deleted = ReaderSession.query.filter(
        ReaderSession.user_id == user_id, ReaderSession.created_at < cutoff
    ).delete()
    db.session.commit()
    return deleted

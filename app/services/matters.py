"""
Matters and their records: the working core of the application.

Everything else in this codebase exists to serve what happens here.
Summarising a recording is useful; being able to produce that recording
eight months later, prove it has not been touched since the day it was
made, and hand someone a pack they can check without trusting you, is
the part that cannot be obtained anywhere else.

Order of operations when a record is entered
--------------------------------------------
1. The original is written to disk and hashed. The hash is of the bytes
   as received, before any processing, because what gets disputed is the
   original, not our transcription of it.
2. Circumstances are captured: entry time, account, device, file size,
   media type, the note the person wrote at the time.
3. Both are sealed into a ledger entry that commits to the previous one.

Step 3 is irreversible by design. There is no edit path for a sealed
record, and that is not an oversight: a record that can be revised
afterwards is exactly the thing the other side will accuse you of
having revised. A correction is entered as a new record that refers to
the earlier one, which is how correction has always worked in any
serious register.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from ..core import ledger
from ..extensions import db
from ..models import HistoryEntry, Matter, MatterRecord, MatterSeal, SourceAsset
from . import source_store

logger = logging.getLogger(__name__)

MATTER_KINDS = [
    ("tenancy", "Tenancy or property"),
    ("contract", "Contract or freelance work"),
    ("employment", "Employment or workplace"),
    ("insurance", "Insurance claim"),
    ("purchase", "Purchase or service dispute"),
    ("other", "Other"),
]

RECORD_KINDS = [
    ("audio", "Recording"),
    ("video", "Video"),
    ("image", "Photograph"),
    ("pdf", "Document"),
    ("text", "Written statement"),
    ("url", "Web page"),
    ("note", "Note"),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str:
    aware = _as_aware(value)
    return aware.isoformat() if aware else ""


# ---------------------------------------------------------------------------
# Matters
# ---------------------------------------------------------------------------


def next_reference(user_id: int) -> str:
    """Sequential per account: M-0001, M-0002. Gaps are fine (a deleted
    matter leaves one) but numbers are never reused, because a reference
    quoted in a letter must keep meaning the same thing."""
    highest = 0
    for row in Matter.query.filter_by(user_id=user_id).all():
        try:
            highest = max(highest, int((row.reference or "M-0").split("-")[-1]))
        except (ValueError, IndexError):
            continue
    return f"M-{highest + 1:04d}"


def create_matter(user_id: int, *, title: str, kind: str = "other", counterparty: str = "", description: str = "") -> Matter:
    matter = Matter(
        user_id=user_id,
        reference=next_reference(user_id),
        title=(title or "Untitled matter").strip()[:300],
        kind=kind if kind in {k for k, _ in MATTER_KINDS} else "other",
        counterparty=(counterparty or "").strip()[:300],
        description=(description or "").strip(),
    )
    db.session.add(matter)
    db.session.commit()
    return matter


def get_matter(user_id: int, matter_id: int) -> Matter | None:
    return Matter.query.filter_by(id=matter_id, user_id=user_id).first()


def list_matters(user_id: int, *, include_closed: bool = True) -> list[dict]:
    query = Matter.query.filter_by(user_id=user_id)
    if not include_closed:
        query = query.filter(Matter.status != "closed")
    matters = query.order_by(Matter.updated_at.desc()).all()

    out = []
    for matter in matters:
        record_count = MatterRecord.query.filter_by(matter_id=matter.id).count()
        last_seal = (
            MatterSeal.query.filter_by(matter_id=matter.id)
            .order_by(MatterSeal.id.desc())
            .first()
        )
        out.append(
            {
                **matter_dict(matter),
                "record_count": record_count,
                "sealed_count": last_seal.record_count if last_seal else 0,
                "unsealed_count": record_count - (last_seal.record_count if last_seal else 0),
                "last_sealed_at": _iso(last_seal.sealed_at) if last_seal else None,
            }
        )
    return out


def matter_dict(matter: Matter) -> dict:
    return {
        "id": matter.id,
        "reference": matter.reference,
        "title": matter.title,
        "kind": matter.kind,
        "kind_label": dict(MATTER_KINDS).get(matter.kind, "Other"),
        "counterparty": matter.counterparty or "",
        "description": matter.description or "",
        "status": matter.status,
        "opened_at": _iso(matter.opened_at),
        "updated_at": _iso(matter.updated_at),
    }


def update_matter(user_id: int, matter_id: int, **fields) -> Matter | None:
    matter = get_matter(user_id, matter_id)
    if matter is None:
        return None
    for key in ("title", "counterparty", "description"):
        if key in fields:
            setattr(matter, key, (fields[key] or "").strip())
    if "kind" in fields and fields["kind"] in {k for k, _ in MATTER_KINDS}:
        matter.kind = fields["kind"]
    if "status" in fields and fields["status"] in {"open", "closed"}:
        matter.status = fields["status"]
        matter.closed_at = _now() if fields["status"] == "closed" else None
    db.session.commit()
    return matter


def delete_matter(user_id: int, matter_id: int) -> bool:
    """Remove a matter and its records.

    The stored originals go too. A partial delete that leaves the files
    would be worse than useless -- someone deleting a matter is usually
    doing it because the contents are sensitive.
    """
    matter = get_matter(user_id, matter_id)
    if matter is None:
        return False
    for record in MatterRecord.query.filter_by(matter_id=matter_id, user_id=user_id).all():
        if record.asset_id:
            source_store.delete_asset(record.asset_id, user_id)
        db.session.delete(record)
    MatterSeal.query.filter_by(matter_id=matter_id, user_id=user_id).delete()
    db.session.delete(matter)
    db.session.commit()
    return True


# ---------------------------------------------------------------------------
# Entering records
# ---------------------------------------------------------------------------


def _last_entry(matter_id: int) -> ledger.LedgerEntry | None:
    row = (
        MatterRecord.query.filter_by(matter_id=matter_id)
        .order_by(MatterRecord.sequence.desc())
        .first()
    )
    if row is None:
        return None
    return ledger.LedgerEntry(
        sequence=row.sequence,
        record_id=row.record_uid,
        content_hash=row.content_hash,
        metadata=json.loads(row.metadata_json or "{}"),
        prev_hash=row.prev_hash,
        entry_hash=row.entry_hash,
    )


def add_record(
    user_id: int,
    matter_id: int,
    *,
    kind: str,
    note: str = "",
    file_bytes: bytes | None = None,
    filename: str = "",
    mime_type: str = "",
    text: str = "",
    url: str = "",
    occurred_at: datetime | None = None,
    entry_id: int | None = None,
    device: str = "",
) -> dict | None:
    """Seal one item into a matter.

    Returns the record as a dict, or None if the matter isn't the
    caller's. Everything that can fail cheaply is checked before the
    chain is touched, because a half-written ledger entry is far worse
    than a rejected upload.
    """
    matter = get_matter(user_id, matter_id)
    if matter is None:
        return None

    kind = kind if kind in {k for k, _ in RECORD_KINDS} else "note"

    asset = None
    if file_bytes:
        content_hash = ledger.hash_bytes(file_bytes)
        asset = source_store.store_bytes(
            user_id, file_bytes, kind=kind, filename=filename,
            mime_type=mime_type, entry_id=entry_id, original_url=url,
        )
    else:
        # Text, URLs and plain notes have no file; the content itself is
        # what gets hashed so they are sealed on exactly the same terms.
        payload = text or url or note
        content_hash = ledger.hash_text(payload)

    record_uid = uuid.uuid4().hex
    entered_at = _now()

    # Everything in here is inside the hash. Anything that might later be
    # edited must stay out of it -- see the module docstring.
    metadata = {
        "kind": kind,
        "note": (note or "").strip(),
        "filename": filename or "",
        "mime_type": mime_type or "",
        "byte_size": len(file_bytes) if file_bytes else 0,
        "url": url or "",
        "text_length": len(text or ""),
        "entered_at": entered_at.isoformat(),
        "occurred_at": _iso(occurred_at) or entered_at.isoformat(),
        "account": str(user_id),
        "device": (device or "")[:200],
        "matter_reference": matter.reference,
    }

    previous = _last_entry(matter_id)
    sealed = ledger.append_entry(
        previous, record_id=record_uid, content_hash=content_hash, metadata=metadata
    )

    record = MatterRecord(
        user_id=user_id,
        matter_id=matter_id,
        entry_id=entry_id,
        asset_id=asset.id if asset else None,
        record_uid=record_uid,
        sequence=sealed.sequence,
        kind=kind,
        note=(note or "").strip(),
        occurred_at=_as_aware(occurred_at) or entered_at,
        content_hash=content_hash,
        metadata_json=json.dumps(metadata, sort_keys=True),
        prev_hash=sealed.prev_hash,
        entry_hash=sealed.entry_hash,
        captured_at=entered_at,
    )
    db.session.add(record)
    matter.updated_at = entered_at
    db.session.commit()

    return record_dict(record)


def record_dict(record: MatterRecord, *, with_asset: bool = True) -> dict:
    data = {
        "id": record.id,
        "matter_id": record.matter_id,
        "record_uid": record.record_uid,
        "sequence": record.sequence,
        "kind": record.kind,
        "kind_label": dict(RECORD_KINDS).get(record.kind, "Record"),
        "note": record.note or "",
        "entry_id": record.entry_id,
        "asset_id": record.asset_id,
        "content_hash": record.content_hash,
        "entry_hash": record.entry_hash,
        "prev_hash": record.prev_hash,
        "fingerprint": ledger.fingerprint_short(record.entry_hash),
        "captured_at": _iso(record.captured_at),
        "occurred_at": _iso(record.occurred_at),
    }
    if with_asset and record.asset_id:
        asset = SourceAsset.query.filter_by(id=record.asset_id, user_id=record.user_id).first()
        if asset is not None:
            data["asset"] = source_store.asset_dict(asset)
    if record.entry_id:
        entry = HistoryEntry.query.filter_by(id=record.entry_id).first()
        if entry is not None:
            data["summary"] = entry.summary
    return data


def list_records(user_id: int, matter_id: int) -> list[dict]:
    records = (
        MatterRecord.query.filter_by(matter_id=matter_id, user_id=user_id)
        .order_by(MatterRecord.sequence.asc())
        .all()
    )
    return [record_dict(r) for r in records]


def get_record(user_id: int, record_id: int) -> MatterRecord | None:
    return MatterRecord.query.filter_by(id=record_id, user_id=user_id).first()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_matter(user_id: int, matter_id: int, *, check_files: bool = True) -> dict:
    """Recompute the chain and compare the stored originals against it.

    `check_files` re-hashes every file on disk. That is the slow part --
    it reads every byte -- but it is also the check that catches a file
    swapped underneath us, which the chain by itself cannot see.
    """
    matter = get_matter(user_id, matter_id)
    if matter is None:
        return {"valid": False, "summary": "No such matter.", "problems": [], "entries": 0}

    rows = (
        MatterRecord.query.filter_by(matter_id=matter_id, user_id=user_id)
        .order_by(MatterRecord.sequence.asc())
        .all()
    )
    entries = [
        ledger.LedgerEntry(
            sequence=r.sequence,
            record_id=r.record_uid,
            content_hash=r.content_hash,
            metadata=json.loads(r.metadata_json or "{}"),
            prev_hash=r.prev_hash,
            entry_hash=r.entry_hash,
        )
        for r in rows
    ]

    content_hashes: dict[str, str] | None = None
    if check_files:
        content_hashes = {}
        for row in rows:
            if not row.asset_id:
                # No file: the sealed hash is of the text itself, which
                # lives in the ledger. Nothing on disk to re-read.
                content_hashes[row.record_uid] = row.content_hash
                continue
            asset = SourceAsset.query.filter_by(id=row.asset_id, user_id=user_id).first()
            path = source_store.absolute_path(asset) if asset else None
            if path is None:
                continue  # verify_chain reports this as a missing file
            try:
                with open(path, "rb") as handle:
                    content_hashes[row.record_uid] = ledger.hash_stream(
                        iter(lambda: handle.read(1024 * 1024), b"")
                    )
            except OSError:
                continue

    result = ledger.verify_chain(entries, content_hashes=content_hashes)
    result["matter"] = matter_dict(matter)
    result["merkle_root"] = ledger.merkle_root([e.entry_hash for e in entries])
    result["checked_files"] = check_files
    result["verified_at"] = _iso(_now())
    return result


# ---------------------------------------------------------------------------
# Sealing
# ---------------------------------------------------------------------------


def seal_matter(user_id: int, matter_id: int) -> dict | None:
    """Fix the current state of a matter and produce the text to publish."""
    matter = get_matter(user_id, matter_id)
    if matter is None:
        return None

    rows = (
        MatterRecord.query.filter_by(matter_id=matter_id, user_id=user_id)
        .order_by(MatterRecord.sequence.asc())
        .all()
    )
    if not rows:
        return None

    entries = [
        ledger.LedgerEntry(
            sequence=r.sequence,
            record_id=r.record_uid,
            content_hash=r.content_hash,
            metadata=json.loads(r.metadata_json or "{}"),
            prev_hash=r.prev_hash,
            entry_hash=r.entry_hash,
        )
        for r in rows
    ]

    sealed_at = _now()
    manifest = ledger.seal_manifest(
        entries,
        matter={"title": matter.title, "reference": matter.reference, "kind": matter.kind,
                "counterparty": matter.counterparty or ""},
        sealed_at=sealed_at.isoformat(),
    )
    anchor = ledger.anchor_text(manifest)

    seal = MatterSeal(
        user_id=user_id,
        matter_id=matter_id,
        record_count=len(entries),
        head_hash=manifest["head_hash"],
        merkle_root=manifest["merkle_root"],
        manifest_json=json.dumps(manifest),
        anchor_text=anchor,
        sealed_at=sealed_at,
    )
    db.session.add(seal)
    matter.updated_at = sealed_at
    db.session.commit()

    return seal_dict(seal)


def seal_dict(seal: MatterSeal) -> dict:
    return {
        "id": seal.id,
        "matter_id": seal.matter_id,
        "record_count": seal.record_count,
        "head_hash": seal.head_hash,
        "merkle_root": seal.merkle_root,
        "root_fingerprint": ledger.fingerprint_short(seal.merkle_root),
        "anchor_text": seal.anchor_text or "",
        "anchored_note": seal.anchored_note or "",
        "sealed_at": _iso(seal.sealed_at),
    }


def list_seals(user_id: int, matter_id: int) -> list[dict]:
    seals = (
        MatterSeal.query.filter_by(matter_id=matter_id, user_id=user_id)
        .order_by(MatterSeal.id.desc())
        .all()
    )
    return [seal_dict(s) for s in seals]


def latest_seal(user_id: int, matter_id: int) -> dict | None:
    seal = (
        MatterSeal.query.filter_by(matter_id=matter_id, user_id=user_id)
        .order_by(MatterSeal.id.desc())
        .first()
    )
    return seal_dict(seal) if seal else None


def record_anchor_note(user_id: int, seal_id: int, note: str) -> bool:
    """Where the user says they published the seal -- emailed to a
    solicitor, posted publicly, sent to themselves. Recorded because in
    six months nobody remembers, and the value of an anchor is entirely
    in being able to point at it."""
    seal = MatterSeal.query.filter_by(id=seal_id, user_id=user_id).first()
    if seal is None:
        return False
    seal.anchored_note = (note or "").strip()[:500]
    db.session.commit()
    return True


def matter_statistics(user_id: int, matter_id: int) -> dict:
    rows = MatterRecord.query.filter_by(matter_id=matter_id, user_id=user_id).all()
    by_kind: dict[str, int] = {}
    total_bytes = 0
    for row in rows:
        by_kind[row.kind] = by_kind.get(row.kind, 0) + 1
        if row.asset_id:
            asset = SourceAsset.query.filter_by(id=row.asset_id).first()
            if asset:
                total_bytes += asset.byte_size or 0

    first = min((r.occurred_at for r in rows if r.occurred_at), default=None)
    last = max((r.occurred_at for r in rows if r.occurred_at), default=None)
    return {
        "record_count": len(rows),
        "by_kind": by_kind,
        "total_bytes": total_bytes,
        "first_event": _iso(first),
        "last_event": _iso(last),
    }

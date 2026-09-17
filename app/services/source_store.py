"""
Keeping the original input, so a summary is never a dead end.

The problem this fixes
----------------------
The app used to extract text from whatever you gave it, summarize that,
and drop the original on the floor. Navigate away and the recording you
just made was gone; the PDF you uploaded was gone; there was no way to
listen again, re-read the source, check what the summary left out, or
re-run it at a different length. You were left holding a paragraph of
output with no way back to the input -- which is a strange thing for a
tool whose whole job is helping you handle documents.

Now every input is written to disk the moment it arrives and linked to
its entry, so `/entry/<id>` can always show the summary and the real
source side by side, with a player for audio and video, the file itself
for PDFs and images, and editable text for everything else.

Storage layout
--------------
    instance/sources/<user_id>/<yyyy-mm>/<sha256[:16]>-<safe-filename>

Files live on disk rather than in the database because a 200MB video in
a BLOB column makes every backup, query plan and connection pool worse,
for no benefit. Content-addressing by SHA-256 means uploading the same
file twice costs one copy, and gives a cheap integrity check.

Paths are stored relative to the sources root. An absolute path baked
into a row would break the moment the app moved directory or was
deployed somewhere else -- and we intend to deploy this.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app

from ..extensions import db
from ..models import HistoryEntry, SourceAsset

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Anything bigger than this is stored but never read back into memory
# whole -- we stream it instead. Matters for video.
_INLINE_READ_LIMIT = 8 * 1024 * 1024


def sources_root() -> Path:
    root = Path(current_app.instance_path) / "sources"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_name(filename: str) -> str:
    """Filenames come from users and from the internet, so they are
    hostile input: they can contain path separators, `..`, control
    characters, or be 4000 bytes long. Reduce to a conservative charset
    and a sane length rather than trusting any of it."""
    name = _UNSAFE.sub("_", (filename or "source").strip()) or "source"
    name = name.lstrip(".") or "source"
    if len(name) > 90:
        stem, dot, ext = name.rpartition(".")
        name = (stem[:80] + ("." + ext if dot else "")) if stem else name[:90]
    return name


def store_bytes(
    user_id: int,
    data: bytes,
    *,
    kind: str,
    filename: str = "",
    mime_type: str = "",
    entry_id: int | None = None,
    original_url: str = "",
) -> SourceAsset | None:
    """Persist raw bytes and record them. Returns None rather than
    raising if storage fails: losing the ability to replay a source is
    bad, but failing the summarize request the user is actually waiting
    on would be worse."""
    if not data:
        return None
    try:
        digest = hashlib.sha256(data).hexdigest()
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        directory = sources_root() / str(user_id) / month
        directory.mkdir(parents=True, exist_ok=True)

        safe = _safe_name(filename or f"{kind}.bin")
        path = directory / f"{digest[:16]}-{safe}"
        if not path.exists():
            path.write_bytes(data)

        asset = SourceAsset(
            user_id=user_id,
            entry_id=entry_id,
            kind=kind,
            filename=filename or safe,
            storage_path=str(path.relative_to(sources_root())),
            mime_type=mime_type or "",
            byte_size=len(data),
            sha256=digest,
            original_url=original_url or "",
        )
        db.session.add(asset)
        db.session.commit()
        return asset
    except Exception as exc:
        logger.warning("Could not store source asset: %s", exc)
        db.session.rollback()
        return None


def store_reference(
    user_id: int,
    *,
    kind: str,
    original_url: str = "",
    filename: str = "",
    entry_id: int | None = None,
) -> SourceAsset | None:
    """Record a source that has no bytes of its own -- a URL or pasted
    text, where `HistoryEntry.source_text` already holds the content."""
    try:
        asset = SourceAsset(
            user_id=user_id,
            entry_id=entry_id,
            kind=kind,
            filename=filename or "",
            storage_path="",
            original_url=original_url or "",
            byte_size=0,
        )
        db.session.add(asset)
        db.session.commit()
        return asset
    except Exception as exc:
        logger.warning("Could not record source reference: %s", exc)
        db.session.rollback()
        return None


def attach_to_entry(asset_id: int, entry_id: int, user_id: int) -> None:
    """Link a source captured before the summary existed to the entry it
    produced. Recordings are stored the instant they're made -- before
    we know whether summarization will even succeed -- so this second
    step is what connects the two."""
    asset = SourceAsset.query.filter_by(id=asset_id, user_id=user_id).first()
    if asset is not None:
        asset.entry_id = entry_id
        db.session.commit()


def assets_for_entry(entry_id: int, user_id: int) -> list[SourceAsset]:
    return (
        SourceAsset.query.filter_by(entry_id=entry_id, user_id=user_id)
        .order_by(SourceAsset.id.asc())
        .all()
    )


def get_asset(asset_id: int, user_id: int) -> SourceAsset | None:
    return SourceAsset.query.filter_by(id=asset_id, user_id=user_id).first()


def absolute_path(asset: SourceAsset) -> Path | None:
    if not asset or not asset.storage_path:
        return None
    path = sources_root() / asset.storage_path
    return path if path.exists() else None


def asset_dict(asset: SourceAsset) -> dict:
    path = absolute_path(asset)
    return {
        "id": asset.id,
        "kind": asset.kind,
        "filename": asset.filename,
        "mime_type": asset.mime_type,
        "byte_size": asset.byte_size,
        "sha256": asset.sha256,
        "original_url": asset.original_url,
        "available": path is not None,
        "created_at": asset.created_at.isoformat() if asset.created_at else None,
    }


def orphan_assets(user_id: int, limit: int = 50) -> list[SourceAsset]:
    """Captures that never became an entry -- a recording made but never
    summarized, or one whose summarization failed. Surfacing these is
    the difference between "my recording vanished" and "your recording
    is here, want to try again?"."""
    return (
        SourceAsset.query.filter_by(user_id=user_id, entry_id=None)
        .order_by(SourceAsset.id.desc())
        .limit(limit)
        .all()
    )


def storage_usage(user_id: int) -> dict:
    """What this account is actually using on disk, so the storage page
    isn't a guess."""
    rows = SourceAsset.query.filter_by(user_id=user_id).all()
    total = sum(r.byte_size or 0 for r in rows)
    by_kind: dict[str, dict] = {}
    for row in rows:
        bucket = by_kind.setdefault(row.kind, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += row.byte_size or 0
    return {
        "total_bytes": total,
        "total_files": len(rows),
        "by_kind": by_kind,
    }


def delete_asset(asset_id: int, user_id: int) -> bool:
    """Remove an asset and its file. The row is authoritative: if the
    file is already gone we still drop the row, so a half-deleted state
    can't linger and keep claiming space that isn't used."""
    asset = SourceAsset.query.filter_by(id=asset_id, user_id=user_id).first()
    if asset is None:
        return False
    path = absolute_path(asset)
    if path is not None:
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not delete file for asset %s: %s", asset_id, exc)
    db.session.delete(asset)
    db.session.commit()
    return True


def delete_assets_for_entry(entry_id: int, user_id: int) -> int:
    count = 0
    for asset in assets_for_entry(entry_id, user_id):
        if delete_asset(asset.id, user_id):
            count += 1
    return count


def purge_user_storage(user_id: int) -> int:
    """Delete every stored source for an account. Used by "delete all my
    data", which must actually delete the files and not merely forget
    the rows pointing at them."""
    directory = sources_root() / str(user_id)
    removed = SourceAsset.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    return removed


def readable_text_for_entry(entry: HistoryEntry) -> str:
    """The text we'd re-summarize if the user asks for a different
    detail level. Stored source text is preferred; the summary is the
    fallback so the action is never simply unavailable."""
    return (entry.source_text or "").strip() or (entry.summary or "")


def human_size(num_bytes: int | None) -> str:
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"

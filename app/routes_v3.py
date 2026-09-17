"""
Matters: the consolidated working surface.

This blueprint replaces a scattered set of feature pages with one
workflow. A person opening this application has a situation they are
documenting -- a tenancy, a contract, a grievance -- and everything they
need sits inside that matter: add a record, read what is there, check it
is intact, export it.

The older feature pages still exist and still work, but they are no
longer the way in. Summarising, searching and chronology are useful
things to do *to* a matter; they were never a reason to open the
application.
"""
from __future__ import annotations

import io
import json
import logging
import mimetypes
from datetime import datetime, timezone

from flask import (
    Blueprint,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required

from .core import ledger
from .models import Job
from .services import evidence_pack, insights, matters as matters_service, source_store

logger = logging.getLogger(__name__)
bp = Blueprint("matters", __name__)

# Uploads land in memory before being hashed and written. Flask's
# MAX_CONTENT_LENGTH is the real ceiling; this is the friendlier message
# for the common case of a long video.
_LARGE_FILE_HINT = 200 * 1024 * 1024


def _json_body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _owned_matter(matter_id: int):
    matter = matters_service.get_matter(current_user.id, matter_id)
    if matter is None:
        abort(404)
    return matter


def _parse_when(value: str) -> datetime | None:
    """Accept what a date input actually sends, and fail quietly.

    A rejected timestamp must never block a record being entered -- the
    record matters far more than the precision of its 'when', and the
    entry time is always captured regardless.
    """
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ===========================================================================
# Matters
# ===========================================================================


@bp.get("/matters")
@login_required
def matter_list():
    return render_template(
        "matters.html",
        matters=matters_service.list_matters(current_user.id),
        kinds=matters_service.MATTER_KINDS,
    )


@bp.post("/matters")
@login_required
def matter_create():
    matter = matters_service.create_matter(
        current_user.id,
        title=request.form.get("title", ""),
        kind=request.form.get("kind", "other"),
        counterparty=request.form.get("counterparty", ""),
        description=request.form.get("description", ""),
    )
    return redirect(url_for("matters.matter_detail", matter_id=matter.id))


@bp.get("/matter/<int:matter_id>")
@login_required
def matter_detail(matter_id: int):
    matter = _owned_matter(matter_id)
    records = matters_service.list_records(current_user.id, matter_id)

    # Chronology and inconsistencies are fetched by the page after load,
    # from their own endpoints. Scanning every record's text on a page
    # render would make opening a large matter slow for two panels the
    # user may not look at.
    return render_template(
        "matter.html",
        matter=matters_service.matter_dict(matter),
        records=records,
        stats=matters_service.matter_statistics(current_user.id, matter_id),
        seal=matters_service.latest_seal(current_user.id, matter_id),
        seals=matters_service.list_seals(current_user.id, matter_id),
        record_kinds=matters_service.RECORD_KINDS,
        kinds=matters_service.MATTER_KINDS,
    )


@bp.post("/matter/<int:matter_id>/update")
@login_required
def matter_update(matter_id: int):
    _owned_matter(matter_id)
    matters_service.update_matter(
        current_user.id,
        matter_id,
        title=request.form.get("title"),
        counterparty=request.form.get("counterparty"),
        description=request.form.get("description"),
        kind=request.form.get("kind"),
    )
    return redirect(url_for("matters.matter_detail", matter_id=matter_id))


@bp.post("/matter/<int:matter_id>/delete")
@login_required
def matter_delete(matter_id: int):
    _owned_matter(matter_id)
    if (request.form.get("confirm") or "").strip().lower() != "delete":
        return redirect(url_for("matters.matter_detail", matter_id=matter_id))
    matters_service.delete_matter(current_user.id, matter_id)
    return redirect(url_for("matters.matter_list"))


# ===========================================================================
# Entering records
# ===========================================================================


@bp.post("/matter/<int:matter_id>/records")
@login_required
def record_add(matter_id: int):
    """Enter one record.

    Accepts a form post (the no-JavaScript path) or JSON. The file is
    read once, hashed, and written; there is no route by which it
    reaches storage unhashed.
    """
    _owned_matter(matter_id)

    kind = request.form.get("kind", "note")
    note = request.form.get("note", "")
    occurred_at = _parse_when(request.form.get("occurred_at", ""))
    text = request.form.get("text", "")
    url_value = request.form.get("url", "")

    file_bytes = None
    filename = ""
    mime_type = ""
    upload = request.files.get("file")
    if upload is not None and upload.filename:
        file_bytes = upload.read()
        filename = upload.filename
        mime_type = upload.mimetype or mimetypes.guess_type(filename)[0] or ""
        if len(file_bytes) > _LARGE_FILE_HINT:
            return render_template(
                "error.html",
                message="That file is larger than the current upload limit. Split it, or compress it, and enter it again.",
            ), 413

    if not any([file_bytes, text.strip(), url_value.strip(), note.strip()]):
        return redirect(url_for("matters.matter_detail", matter_id=matter_id, error="empty"))

    matters_service.add_record(
        current_user.id,
        matter_id,
        kind=kind,
        note=note,
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        text=text,
        url=url_value,
        occurred_at=occurred_at,
        device=(request.headers.get("User-Agent") or "")[:200],
    )
    return redirect(url_for("matters.matter_detail", matter_id=matter_id, added="1"))


@bp.post("/api/matter/<int:matter_id>/records")
@login_required
def api_record_add(matter_id: int):
    _owned_matter(matter_id)
    body = _json_body()
    record = matters_service.add_record(
        current_user.id,
        matter_id,
        kind=body.get("kind", "note"),
        note=body.get("note", ""),
        text=body.get("text", ""),
        url=body.get("url", ""),
        occurred_at=_parse_when(body.get("occurred_at", "")),
        device=(request.headers.get("User-Agent") or "")[:200],
    )
    if record is None:
        return jsonify({"error": "No such matter."}), 404
    return jsonify({"record": record})


@bp.get("/api/matter/<int:matter_id>/records")
@login_required
def api_records(matter_id: int):
    _owned_matter(matter_id)
    return jsonify({"records": matters_service.list_records(current_user.id, matter_id)})


@bp.get("/record/<int:record_id>/file")
@login_required
def record_file(record_id: int):
    """Serve the original.

    Every record page links here rather than embedding a path, so there
    is exactly one ownership check and no way to construct a path from
    the request.
    """
    record = matters_service.get_record(current_user.id, record_id)
    if record is None or not record.asset_id:
        abort(404)
    asset = source_store.get_asset(record.asset_id, current_user.id)
    path = source_store.absolute_path(asset) if asset else None
    if path is None:
        abort(404)
    guessed = asset.mime_type or mimetypes.guess_type(asset.filename or "")[0] or "application/octet-stream"
    return send_file(path, mimetype=guessed, as_attachment=False,
                     download_name=asset.filename or f"record-{record.sequence}")


# ===========================================================================
# Verification and sealing
# ===========================================================================


@bp.get("/matter/<int:matter_id>/verify")
@login_required
def matter_verify(matter_id: int):
    _owned_matter(matter_id)
    result = matters_service.verify_matter(current_user.id, matter_id, check_files=True)
    return render_template(
        "verify.html",
        result=result,
        matter=result.get("matter"),
        seal=matters_service.latest_seal(current_user.id, matter_id),
        matter_id=matter_id,
    )


@bp.get("/api/matter/<int:matter_id>/verify")
@login_required
def api_verify(matter_id: int):
    _owned_matter(matter_id)
    return jsonify(matters_service.verify_matter(current_user.id, matter_id, check_files=True))


@bp.post("/matter/<int:matter_id>/seal")
@login_required
def matter_seal(matter_id: int):
    _owned_matter(matter_id)
    seal = matters_service.seal_matter(current_user.id, matter_id)
    if seal is None:
        return redirect(url_for("matters.matter_detail", matter_id=matter_id, error="nothing_to_seal"))
    return redirect(url_for("matters.matter_detail", matter_id=matter_id, sealed=seal["id"]))


@bp.post("/api/seal/<int:seal_id>/anchor")
@login_required
def api_seal_anchor(seal_id: int):
    note = (_json_body().get("note") or "").strip()
    if not matters_service.record_anchor_note(current_user.id, seal_id, note):
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True})


@bp.get("/matter/<int:matter_id>/pack")
@login_required
def matter_pack(matter_id: int):
    """Download the exportable pack."""
    _owned_matter(matter_id)
    built = evidence_pack.build_pack(current_user.id, matter_id)
    if built is None:
        return redirect(url_for("matters.matter_detail", matter_id=matter_id, error="nothing_to_export"))
    data, filename = built
    return send_file(
        io.BytesIO(data),
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


# ===========================================================================
# Work in progress
#
# Long jobs previously died with the page that started them, so leaving
# a page mid-upload lost the work with no indication anything had been
# running. Jobs are rows in the database and always were; what was
# missing was any way for a page to find out about jobs it did not
# itself start.
# ===========================================================================


@bp.get("/api/jobs/active")
@login_required
def api_jobs_active():
    rows = (
        Job.query.filter(Job.user_id == current_user.id, Job.status.in_(["running", "queued"]))
        .order_by(Job.created_at.desc())
        .limit(12)
        .all()
    )

    jobs = []
    for row in rows:
        try:
            steps = json.loads(row.progress or "[]")
        except json.JSONDecodeError:
            steps = []
        jobs.append(
            {
                "id": row.id,
                "kind": row.kind or "job",
                "status": row.status,
                "steps": steps,
                "latest": steps[-1] if steps else "Starting...",
                "step_count": len(steps),
                "started_at": row.created_at.isoformat() if row.created_at else "",
            }
        )

    recent_done = (
        Job.query.filter(Job.user_id == current_user.id, Job.status == "done")
        .order_by(Job.updated_at.desc())
        .limit(3)
        .all()
    )
    finished = []
    for row in recent_done:
        try:
            result = json.loads(row.result or "{}")
        except json.JSONDecodeError:
            result = {}
        finished.append(
            {
                "id": row.id,
                "kind": row.kind or "job",
                "doc_id": result.get("doc_id"),
                "finished_at": row.updated_at.isoformat() if row.updated_at else "",
            }
        )

    return jsonify({"active": jobs, "recent": finished})


# ===========================================================================
# Reading a matter: chronology and inconsistencies, scoped to one matter
# ===========================================================================


@bp.get("/api/matter/<int:matter_id>/chronology")
@login_required
def api_matter_chronology(matter_id: int):
    """Dates mentioned across a matter's records, in order.

    In a dispute the sequence of events is usually the whole argument,
    and reconstructing it by hand from twenty records is exactly the
    tedious work a machine should do.
    """
    _owned_matter(matter_id)
    records = matters_service.list_records(current_user.id, matter_id)
    documents = []
    for record in records:
        text = (record.get("summary") or "") + "\n" + (record.get("note") or "")
        if not text.strip():
            continue
        documents.append(
            {
                "id": record["id"],
                "title": record["note"] or record["kind_label"],
                "text": text,
                "created_at": record["occurred_at"] or record["captured_at"],
            }
        )

    if not documents:
        return jsonify({"events": [], "total": 0})

    from .core import temporal

    timeline = temporal.build_timeline(documents)
    timeline["events"] = [e for e in timeline.get("events", []) if e.get("confidence", 0) >= 0.4]
    timeline["total"] = len(timeline["events"])
    return jsonify(timeline)


@bp.get("/api/matter/<int:matter_id>/inconsistencies")
@login_required
def api_matter_inconsistencies(matter_id: int):
    """Numbers that disagree across a matter's records.

    Two versions of the same figure -- a deposit amount, an invoice
    total, a date of notice -- is often the single most useful thing to
    surface, because it is the point on which an account falls apart.
    """
    _owned_matter(matter_id)
    records = matters_service.list_records(current_user.id, matter_id)
    documents = [
        {
            "id": r["id"],
            "title": r["note"] or r["kind_label"],
            "text": (r.get("summary") or "") + "\n" + (r.get("note") or ""),
        }
        for r in records
        if (r.get("summary") or r.get("note"))
    ]
    if not documents:
        return jsonify({"conflicts": [], "groups": [], "total_quantities": 0})

    from .core import quantities

    return jsonify(quantities.find_contradictions(documents))


@bp.get("/api/matter/<int:matter_id>/summary")
@login_required
def api_matter_summary(matter_id: int):
    """A plain factual overview of the matter, computed locally."""
    _owned_matter(matter_id)
    stats = matters_service.matter_statistics(current_user.id, matter_id)
    verification = matters_service.verify_matter(current_user.id, matter_id, check_files=False)
    seal = matters_service.latest_seal(current_user.id, matter_id)
    return jsonify(
        {
            "statistics": stats,
            "intact": verification["valid"],
            "problems": len(verification.get("problems", [])),
            "sealed": bool(seal),
            "last_sealed_at": seal["sealed_at"] if seal else None,
            "merkle_root": verification.get("merkle_root"),
            "root_fingerprint": ledger.fingerprint_short(verification.get("merkle_root", "")),
        }
    )

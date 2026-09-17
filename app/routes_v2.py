"""
Routes for everything built in the "make this a real product" pass.

Kept as a second blueprint rather than appended to routes.py for a
practical reason: routes.py is the original summarize-and-display app
and is already long. Mixing the new surface into it would make both
harder to read and every future merge harder. These are registered
under the same app with no url_prefix, so URLs stay clean.

Two rules hold throughout:

*Ownership is checked on every single route.* Not on most of them. An
`entry_id` arriving from a URL is an untrusted integer, and the only
safe way to treat it is as a claim to be verified against `current_user`
before anything is read or written.

*Nothing here needs an AI API.* Every feature in this file runs on the
algorithms in app/core/ and the user's own stored data. If the Gemini
key were removed tomorrow, every page below still works.
"""
from __future__ import annotations

import json
import logging
import mimetypes

from flask import (
    Blueprint,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required

from .core import acoustics, reading
from .core import rules as rules_core
from .extensions import db
from .models import HistoryEntry
from .services import (
    automation,
    entry_manager,
    feature_store,
    insights,
    rooms,
    session_guard,
    source_store,
)

logger = logging.getLogger(__name__)
bp = Blueprint("v2", __name__)


def _json_body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _owned_entry(entry_id: int) -> HistoryEntry:
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=current_user.id).first()
    if entry is None:
        abort(404)
    return entry


# ===========================================================================
# The entry workspace -- summary and original source, side by side
# ===========================================================================


@bp.get("/entry/<int:entry_id>")
@login_required
def entry_workspace(entry_id: int):
    entry = _owned_entry(entry_id)
    assets = [source_store.asset_dict(a) for a in source_store.assets_for_entry(entry_id, current_user.id)]
    return render_template(
        "entry.html",
        entry=entry_manager.entry_dict(entry, with_source=True),
        assets=assets,
        all_tags=entry_manager.all_tags(current_user.id),
    )


@bp.get("/source/<int:asset_id>")
@login_required
def serve_source(asset_id: int):
    """Stream a stored original back.

    `send_file` with a path built from a database row is exactly the
    shape of a path-traversal bug, so the row is fetched scoped to the
    current user first and the path is resolved through the same helper
    that wrote it -- the request never supplies a path, only an id we
    look up."""
    asset = source_store.get_asset(asset_id, current_user.id)
    if asset is None:
        abort(404)
    path = source_store.absolute_path(asset)
    if path is None:
        abort(404)
    guessed = asset.mime_type or mimetypes.guess_type(asset.filename or "")[0] or "application/octet-stream"
    return send_file(path, mimetype=guessed, as_attachment=False, download_name=asset.filename or "source")


@bp.get("/api/entry/<int:entry_id>")
@login_required
def api_entry_get(entry_id: int):
    entry = _owned_entry(entry_id)
    return jsonify(entry_manager.entry_dict(entry, with_source=True))


@bp.patch("/api/entry/<int:entry_id>")
@login_required
def api_entry_update(entry_id: int):
    body = _json_body()
    allowed = {k: v for k, v in body.items() if k in {"title", "summary", "notes", "tags", "is_archived", "is_pinned", "source_text"}}
    updated = entry_manager.update_entry(current_user.id, entry_id, **allowed)
    if updated is None:
        return jsonify({"error": "Not found."}), 404
    return jsonify(updated)


@bp.delete("/api/entry/<int:entry_id>")
@login_required
def api_entry_delete(entry_id: int):
    if not entry_manager.delete_entry(current_user.id, entry_id):
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True})


@bp.post("/api/entries/bulk")
@login_required
def api_entries_bulk():
    body = _json_body()
    result = entry_manager.bulk_action(
        current_user.id,
        body.get("entry_ids", []),
        (body.get("action") or "").strip(),
        (body.get("value") or "").strip(),
    )
    return jsonify(result)


# ===========================================================================
# Library -- the manageable knowledge base
# ===========================================================================


@bp.get("/library")
@login_required
def library():
    listing = entry_manager.list_entries(
        current_user.id,
        query=request.args.get("q", ""),
        tag=request.args.get("tag", ""),
        source_type=request.args.get("type", ""),
        archived=request.args.get("archived") == "1",
        limit=int(request.args.get("limit", 60) or 60),
        offset=int(request.args.get("offset", 0) or 0),
    )
    return render_template(
        "library.html",
        listing=listing,
        stats=entry_manager.library_stats(current_user.id),
        tags=entry_manager.all_tags(current_user.id),
        types=entry_manager.source_type_counts(current_user.id),
        storage=source_store.storage_usage(current_user.id),
        orphans=[source_store.asset_dict(a) for a in source_store.orphan_assets(current_user.id, limit=12)],
        query=request.args.get("q", ""),
        active_tag=request.args.get("tag", ""),
        active_type=request.args.get("type", ""),
        showing_archived=request.args.get("archived") == "1",
    )


# ===========================================================================
# Provenance -- "where did this come from?"
# ===========================================================================


@bp.get("/trace")
@login_required
def trace_page():
    entry_id = request.args.get("entry", type=int)
    trace = insights.trace_entry(entry_id, current_user.id) if entry_id else None
    return render_template(
        "trace.html",
        trace=trace,
        entry_id=entry_id,
        documents=feature_store.reader_documents(current_user.id, limit=200),
    )


@bp.get("/api/trace/<int:entry_id>")
@login_required
def api_trace(entry_id: int):
    return jsonify(insights.trace_entry(entry_id, current_user.id))


@bp.post("/api/trace/adhoc")
@login_required
def api_trace_adhoc():
    body = _json_body()
    return jsonify(insights.trace_text(body.get("summary", ""), body.get("source_text", "")))


# ===========================================================================
# Timeline
# ===========================================================================


@bp.get("/timeline")
@login_required
def timeline_page():
    return render_template("timeline.html", timeline=insights.build_timeline(current_user.id))


@bp.get("/api/timeline")
@login_required
def api_timeline():
    return jsonify(insights.build_timeline(current_user.id))


# ===========================================================================
# Contradictions
# ===========================================================================


@bp.get("/contradictions")
@login_required
def contradictions_page():
    return render_template("contradictions.html", report=insights.find_contradictions(current_user.id))


@bp.get("/api/contradictions")
@login_required
def api_contradictions():
    return jsonify(insights.find_contradictions(current_user.id))


# ===========================================================================
# Zero-knowledge vault
# ===========================================================================


@bp.get("/vault")
@login_required
def vault_page():
    return render_template("vault.html", stats=feature_store.vault_stats(current_user.id))


@bp.post("/api/vault/put")
@login_required
def api_vault_put():
    body = _json_body()
    try:
        item = feature_store.vault_put(
            current_user.id,
            body.get("entry_id") or body.get("item_key", ""),
            body.get("ciphertext", ""),
            body.get("iv", ""),
            body.get("salt", ""),
            algo=body.get("algo", "AES-GCM-256"),
            kdf_iterations=int(body.get("kdf_iterations", 600000) or 600000),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "item": item})


@bp.get("/api/vault/list")
@login_required
def api_vault_list():
    return jsonify({"items": feature_store.vault_list(current_user.id)})


@bp.get("/api/vault/get/<item_key>")
@login_required
def api_vault_get(item_key: str):
    item = feature_store.vault_get(current_user.id, item_key)
    if item is None:
        return jsonify({"error": "Not found."}), 404
    return jsonify(item)


@bp.delete("/api/vault/item/<item_key>")
@login_required
def api_vault_delete(item_key: str):
    if not feature_store.vault_delete(current_user.id, item_key):
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True})


# ===========================================================================
# Spatial canvas
# ===========================================================================


@bp.get("/canvas")
@login_required
def canvas_page():
    return render_template("canvas.html")


@bp.get("/api/canvas")
@login_required
def api_canvas_load():
    return jsonify(feature_store.canvas_load(current_user.id))


@bp.post("/api/canvas")
@login_required
def api_canvas_save():
    return jsonify(feature_store.canvas_save(current_user.id, _json_body()))


@bp.get("/api/canvas/items")
@login_required
def api_canvas_items():
    return jsonify({"items": feature_store.canvas_items(current_user.id)})


# ===========================================================================
# Speed reader
# ===========================================================================


@bp.get("/reader")
@login_required
def reader_page():
    return render_template("reader.html")


@bp.get("/api/reader/documents")
@login_required
def api_reader_documents():
    return jsonify({"items": feature_store.reader_documents(current_user.id)})


@bp.get("/api/reader/document/<int:entry_id>")
@login_required
def api_reader_document(entry_id: int):
    document = feature_store.reader_document(
        current_user.id, entry_id, use_summary=request.args.get("summary") == "1"
    )
    if document is None:
        return jsonify({"error": "Nothing to read in that entry."}), 404
    return jsonify(document)


@bp.get("/api/reader/quiz/<int:entry_id>")
@login_required
def api_reader_quiz(entry_id: int):
    entry = _owned_entry(entry_id)
    text = (entry.source_text or "") or (entry.summary or "")
    questions = reading.comprehension_questions(text[:120_000], count=4, seed=entry_id)
    return jsonify({"questions": questions})


@bp.post("/api/reader/session")
@login_required
def api_reader_session():
    body = _json_body()
    comprehension = body.get("comprehension")
    return jsonify(
        feature_store.reader_record_session(
            current_user.id,
            body.get("entry_id"),
            wpm=int(body.get("wpm", 300) or 300),
            words_read=int(body.get("words_read", 0) or 0),
            duration_ms=int(body.get("duration_ms", 0) or 0),
            comprehension=float(comprehension) if comprehension is not None else None,
            completed=bool(body.get("completed")),
        )
    )


@bp.get("/api/reader/stats")
@login_required
def api_reader_stats():
    return jsonify(feature_store.reader_stats(current_user.id))


# ===========================================================================
# Collaboration rooms
# ===========================================================================


@bp.get("/rooms")
@login_required
def rooms_page():
    return render_template(
        "rooms_index.html", documents=feature_store.reader_documents(current_user.id, limit=100)
    )


@bp.get("/room/<room_id>")
@login_required
def room_page(room_id: str):
    room = rooms.get_room(room_id)
    if room is None:
        return render_template("error.html", message="That room has ended or never existed."), 404

    document = None
    if room.get("entry_id"):
        entry = HistoryEntry.query.filter_by(id=room["entry_id"]).first()
        if entry is not None:
            document = {
                "id": entry.id,
                "title": entry.display_title,
                "text": (entry.source_text or "") or (entry.summary or ""),
                "summary": entry.summary or "",
            }
    return render_template("room.html", room=room, document=document)


@bp.post("/api/rooms")
@login_required
def api_rooms_create():
    body = _json_body()
    entry_id = body.get("entry_id")
    if entry_id:
        _owned_entry(int(entry_id))  # refuse to open a room over someone else's document
    room = rooms.create_room(current_user.id, int(entry_id) if entry_id else None, body.get("title", "Reading room"))
    return jsonify({"room": room})


@bp.post("/api/rooms/<room_id>/join")
@login_required
def api_rooms_join(room_id: str):
    body = _json_body()
    try:
        result = rooms.join_room(
            room_id, current_user.id, body.get("display_name") or current_user.display_name or "Guest"
        )
    except rooms.RoomError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@bp.post("/api/rooms/<room_id>/event")
@login_required
def api_rooms_event(room_id: str):
    body = _json_body()
    try:
        event = rooms.publish(
            room_id, body.get("participant_id"), body.get("type", ""), body.get("payload") or {}
        )
    except rooms.RoomError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"event": event})


@bp.post("/api/rooms/<room_id>/leave")
@login_required
def api_rooms_leave(room_id: str):
    rooms.leave_room(room_id, (_json_body().get("participant_id") or ""))
    return jsonify({"ok": True})


@bp.get("/api/rooms/<room_id>/state")
@login_required
def api_rooms_state(room_id: str):
    room = rooms.get_room(room_id)
    if room is None:
        return jsonify({"error": "Not found."}), 404
    return jsonify(
        {
            "room": room,
            "participants": rooms.participants(room_id),
            **rooms.room_state(room_id),
        }
    )


@bp.get("/api/rooms/<room_id>/stream")
@login_required
def api_rooms_stream(room_id: str):
    """Server-sent events.

    SSE rather than WebSockets because it needs no extra dependency, no
    separate server process and survives ordinary HTTP proxies. The
    generator yields heartbeats so intermediaries don't decide an idle
    connection is dead, and `X-Accel-Buffering: no` stops nginx holding
    events in a buffer waiting for more -- which would turn a live
    stream into a batched one.
    """
    participant_id = request.args.get("participant_id", "")
    try:
        last_event_id = int(request.args.get("last_event_id", 0) or 0)
    except ValueError:
        last_event_id = 0

    if rooms.get_room(room_id) is None:
        return jsonify({"error": "Not found."}), 404

    def generate():
        try:
            for frame in rooms.subscribe(room_id, participant_id, last_event_id=last_event_id):
                yield frame if isinstance(frame, str) else rooms.format_sse(frame)
        except rooms.RoomError:
            return
        except GeneratorExit:  # client went away; nothing to clean up here
            return

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ===========================================================================
# Audio fingerprinting
# ===========================================================================


@bp.get("/audio-lab")
@login_required
def audio_lab_page():
    return render_template("audio_lab.html", library=feature_store.fingerprint_library(current_user.id))


@bp.post("/api/audio/fingerprint")
@login_required
def api_audio_fingerprint():
    """Fingerprint PCM samples decoded in the browser.

    The browser does the decoding with WebAudio and sends plain float
    samples, which is why this app can fingerprint an m4a without
    shipping ffmpeg. Arrays are capped so one request can't ask the
    server to allocate an unbounded amount of memory.
    """
    body = _json_body()
    samples = body.get("samples") or []
    sample_rate = int(body.get("sample_rate") or 44100)

    if not samples:
        return jsonify({"error": "No audio samples were sent."}), 400
    if len(samples) > 40_000_000:
        return jsonify({"error": "That recording is too long to analyse in one go."}), 413

    try:
        import numpy as np

        array = np.asarray(samples, dtype=np.float32)
        fingerprint = acoustics.fingerprint(array, sample_rate)
    except Exception as exc:
        logger.warning("Fingerprinting failed: %s", exc)
        return jsonify({"error": "Could not analyse that audio."}), 400

    matches = feature_store.fingerprint_match(current_user.id, fingerprint, top_n=5)
    repeats = acoustics.find_repeats(fingerprint)

    stored = None
    if body.get("save"):
        stored = feature_store.fingerprint_store(
            current_user.id,
            fingerprint,
            label=(body.get("label") or "Recording")[:255],
            entry_id=body.get("entry_id"),
        )

    return jsonify(
        {
            "fingerprint_id": fingerprint["fingerprint_id"],
            "duration": fingerprint["duration"],
            "hash_count": fingerprint["hash_count"],
            "peak_count": fingerprint["peak_count"],
            "matches": matches,
            "repeats": repeats,
            "stored": stored,
        }
    )


@bp.post("/api/audio/spectrogram")
@login_required
def api_audio_spectrogram():
    body = _json_body()
    samples = body.get("samples") or []
    if not samples:
        return jsonify({"error": "No audio samples were sent."}), 400
    try:
        import numpy as np

        array = np.asarray(samples, dtype=np.float32)
        preview = acoustics.spectrogram_preview(array, int(body.get("sample_rate") or 44100))
    except Exception:
        return jsonify({"error": "Could not analyse that audio."}), 400
    return jsonify(preview)


@bp.delete("/api/audio/<int:row_id>")
@login_required
def api_audio_delete(row_id: int):
    if not feature_store.fingerprint_delete(current_user.id, row_id):
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True})


# ===========================================================================
# Automation
# ===========================================================================


@bp.get("/automation")
@login_required
def automation_page():
    return render_template(
        "automation.html",
        rules=automation.list_rules(current_user.id),
        triggers=automation.TRIGGERS,
        action_types=automation.ACTION_TYPES,
        examples=automation.example_rules(),
        runs=automation.recent_runs(current_user.id, limit=25),
    )


@bp.post("/api/automation/validate")
@login_required
def api_automation_validate():
    body = _json_body()
    return jsonify(automation.validate_rule(body.get("condition", ""), body.get("actions", [])))


@bp.post("/api/automation/rule")
@login_required
def api_automation_save():
    body = _json_body()
    try:
        rule = automation.save_rule(
            current_user.id,
            name=body.get("name", ""),
            trigger=body.get("trigger", "entry.created"),
            condition=body.get("condition", ""),
            actions=body.get("actions", []),
            enabled=bool(body.get("enabled", True)),
            rule_id=body.get("id"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "rule": rule})


@bp.delete("/api/automation/rule/<int:rule_id>")
@login_required
def api_automation_delete(rule_id: int):
    if not automation.delete_rule(current_user.id, rule_id):
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True})


@bp.post("/api/automation/rule/<int:rule_id>/toggle")
@login_required
def api_automation_toggle(rule_id: int):
    enabled = bool(_json_body().get("enabled", True))
    if not automation.toggle_rule(current_user.id, rule_id, enabled):
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True, "enabled": enabled})


@bp.post("/api/automation/test")
@login_required
def api_automation_test():
    """Dry-run a rule against a real entry, so the user can see what it
    would do before letting it loose unattended."""
    body = _json_body()
    entry_id = body.get("entry_id")
    if not entry_id:
        return jsonify({"error": "Pick an entry to test against."}), 400
    _owned_entry(int(entry_id))
    return jsonify({"results": automation.run_for_entry(current_user.id, int(entry_id), trigger="manual", dry_run=True)})


@bp.post("/api/automation/run")
@login_required
def api_automation_run():
    body = _json_body()
    entry_id = body.get("entry_id")
    if not entry_id:
        return jsonify({"error": "Pick an entry to run against."}), 400
    _owned_entry(int(entry_id))
    return jsonify({"results": automation.run_for_entry(current_user.id, int(entry_id), trigger="manual")})


@bp.get("/api/automation/language")
@login_required
def api_automation_language():
    """The condition language, described for the in-page help."""
    return jsonify(
        {
            "fields": [
                {"name": "source_type", "example": 'source_type == "pdf"', "about": "text, pdf, article, youtube, audio, image or video"},
                {"name": "title", "example": 'title contains "quarterly"', "about": "The entry's name"},
                {"name": "text", "example": 'text mentions "funding"', "about": "Full source text. `mentions` matches word stems, so funding matches funded"},
                {"name": "word_count", "example": "word_count > 500", "about": "Words in the source"},
                {"name": "tags", "example": 'tags includes "research"', "about": "Tags on the entry"},
                {"name": "domain", "example": 'domain in ["ft.com"]', "about": "Website an article came from"},
                {"name": "archived", "example": "not archived", "about": "True or false"},
            ],
            "operators": ["==", "!=", "<", "<=", ">", ">=", "and", "or", "not", "in", "includes", "contains", "mentions", "matches", "before", "after"],
        }
    )


# ===========================================================================
# Offline sync
# ===========================================================================


@bp.post("/api/sync/push")
@login_required
def api_sync_push():
    body = _json_body()
    return jsonify(feature_store.sync_push(current_user.id, body.get("operations", [])))


@bp.post("/api/sync/pull")
@login_required
def api_sync_pull():
    body = _json_body()
    return jsonify(feature_store.sync_pull(current_user.id, body.get("clock") or {}))


@bp.get("/api/sync/status")
@login_required
def api_sync_status():
    return jsonify(feature_store.sync_status(current_user.id))


@bp.get("/api/sync/state")
@login_required
def api_sync_state():
    return jsonify(feature_store.sync_state(current_user.id))


# ===========================================================================
# Sessions and devices
# ===========================================================================


@bp.get("/settings/security")
@login_required
def security_page():
    return render_template(
        "security.html",
        sessions=session_guard.list_sessions(current_user.id),
        idle_minutes=session_guard.IDLE_TIMEOUT_MINUTES,
        remembered_days=session_guard.REMEMBERED_IDLE_DAYS,
        absolute_days=session_guard.ABSOLUTE_LIFETIME_DAYS,
        vault=feature_store.vault_stats(current_user.id),
        storage=source_store.storage_usage(current_user.id),
    )


@bp.post("/settings/security/revoke/<int:session_id>")
@login_required
def revoke_session(session_id: int):
    session_guard.revoke_session(current_user.id, session_id)
    return redirect(url_for("v2.security_page"))


@bp.post("/settings/security/revoke-others")
@login_required
def revoke_other_sessions():
    session_guard.revoke_all_other_sessions(current_user.id)
    return redirect(url_for("v2.security_page"))


@bp.get("/api/session/heartbeat")
@login_required
def api_session_heartbeat():
    """Lets the page show an honest countdown before an idle logout,
    instead of the user losing half-typed work to a silent redirect."""
    return jsonify({"seconds_remaining": session_guard.seconds_remaining()})


@bp.post("/settings/danger/delete-everything")
@login_required
def delete_everything():
    confirmation = (request.form.get("confirm") or "").strip().lower()
    if confirmation != "delete everything":
        return redirect(url_for("v2.security_page", error="confirm"))
    entry_manager.delete_all_entries(current_user.id)
    return redirect(url_for("v2.security_page", deleted="1"))

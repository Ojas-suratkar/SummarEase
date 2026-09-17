"""
Background-job registry + worker threads. Gives the frontend live
progress on long operations (extraction, TextRank filtering, chunked
Gemini summarization, indexing, saving) without blocking the request or
requiring a full-page reload to see the result.

Deliberately polling-based rather than Server-Sent Events or WebSockets:
Flask's built-in dev server (what this app runs on, via run.py) is not
an async server. A long-lived SSE connection per active tab needs
`threaded=True` and careful handling to avoid starving other requests
under it, for a benefit a human can't actually perceive -- a ~700ms poll
from the browser feels just as instantaneous, works identically under
the dev server, a production WSGI server, or behind a reverse proxy, and
needs no special infrastructure. This is a deliberate trade-off, not an
oversight.

Jobs are kept in a fast in-memory dict for the polling hot path (writing
every progress message to the database would be wasteful), but every
status transition is also written through to the `jobs` table (models.py)
-- so a job that finished right before the dev-server reloader restarted
the process, or a page you refreshed an hour later, still shows its real
result instead of "job not found (it may have expired)", which used to
be true of *every* job after 30 minutes or any restart.

Every classic synchronous route in routes.py (the ones that render a
full result page directly) still works exactly as before and is the
fallback when JavaScript is unavailable; this registry only backs the
`/api/jobs/*` endpoints the frontend uses for the live-progress path.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid

from . import app_context
from ..extensions import db
from ..models import Job

_lock = threading.Lock()
_jobs: dict[str, dict] = {}

_MAX_MEMORY_JOBS = 200


def _evict_memory_locked() -> None:
    if len(_jobs) > _MAX_MEMORY_JOBS:
        oldest = sorted(_jobs.items(), key=lambda kv: kv[1]["created_at"])
        for jid, _ in oldest[: len(_jobs) - _MAX_MEMORY_JOBS]:
            _jobs.pop(jid, None)


def _persist(job_id: str, user_id: int, *, kind: str | None = None, status: str | None = None,
             progress: list[str] | None = None, result: dict | None = None, error: str | None = None,
             create: bool = False) -> None:
    def _write():
        row = Job.query.get(job_id)
        if row is None:
            if not create:
                return
            row = Job(id=job_id, user_id=user_id, kind=kind, status=status or "running")
            db.session.add(row)
        if status is not None:
            row.status = status
        if progress is not None:
            row.progress = json.dumps(progress)
        if result is not None:
            row.result = json.dumps(result)
        if error is not None:
            row.error = error
        db.session.commit()

    try:
        app_context.run(_write)
    except Exception:  # noqa: BLE001 -- persistence is a durability nicety, not the critical path
        traceback.print_exc()


def _push_progress(job_id: str, user_id: int, message: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["progress"].append(message)
            progress_copy = list(job["progress"])
        else:
            progress_copy = None
    if progress_copy is not None:
        _persist(job_id, user_id, progress=progress_copy)


def _run(job_id: str, user_id: int, kind: str, fn, args: tuple, kwargs: dict) -> None:
    try:
        result = app_context.run(
            fn, *args, report=lambda msg: _push_progress(job_id, user_id, msg), **kwargs
        )
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = result
        _persist(job_id, user_id, kind=kind, status="done", result=result)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the client, never swallowed silently
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["error"] = str(exc)
        _persist(job_id, user_id, kind=kind, status="error", error=str(exc))
        traceback.print_exc()


def start_job(user_id: int, fn, *args, kind: str = "", **kwargs) -> str:
    """Start `fn(*args, report=<callable>, **kwargs)` on a background
    thread and return a job_id to poll via `get_job`."""
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "progress": [],
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
        _evict_memory_locked()
    _persist(job_id, user_id, kind=kind, status="running", progress=[], create=True)
    threading.Thread(target=_run, args=(job_id, user_id, kind, fn, args, kwargs), daemon=True).start()
    return job_id


def get_stats(user_id: int) -> dict:
    """Aggregate counts for the ops dashboard (routes.py's /dashboard) --
    how many of this user's background jobs are running/done/errored,
    from the durable table (not just whatever's still warm in memory)."""
    rows = Job.query.filter_by(user_id=user_id).all()
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
    return {"total": len(rows), "by_status": by_status}


def get_job(job_id: str, user_id: int) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return {
                "status": job["status"],
                "progress": list(job["progress"]),
                "result": job["result"],
                "error": job["error"],
            }

    row = Job.query.filter_by(id=job_id, user_id=user_id).first()
    if row is None:
        return None
    return {
        "status": row.status,
        "progress": json.loads(row.progress) if row.progress else [],
        "result": json.loads(row.result) if row.result else None,
        "error": row.error,
    }

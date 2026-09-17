import json

from flask import Blueprint, Response, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from .models import User
from .services import (
    analytics,
    annotations,
    backup,
    cache,
    comparator,
    compose,
    dedup,
    digest,
    export_center,
    gemini_client,
    history_store,
    jobs,
    knowledge_graph,
    perspectives,
    pipeline,
    rag,
    search as search_service,
    sharing,
    spaced_repetition,
    watchlist,
)
from .services.compose import ComposeError
from .services.annotations import AnnotationError
from .services.backup import BackupError
from .services.sharing import SharingError
from .services.export_center import ExportError
from .services.article_extractor import ArticleExtractionError, extract_article_text
from .services.audio_service import AudioServiceError
from .services.credibility_lens import CredibilityError, analyze_credibility
from .services.image_service import ANALYSIS_MODES, ImageServiceError
from .services.pdf_extractor import PdfExtractionError, extract_text as extract_pdf_text
from .services.perspectives import PerspectivesError
from .services.rag import RagError, ask as rag_ask, get_full_text as rag_get_full_text
from .services.spaced_repetition import SpacedRepetitionError
from .services.summarizer import SummarizerError, check_faithfulness, explain_term
from .services.synthesis import SynthesisError, synthesize_sources
from .services.topics import TopicsError, get_topic_clusters
from .services.tts_service import synthesize as synthesize_speech
from .services.video_service import VideoServiceError
from .services.watchlist import WatchlistError
from .services.youtube_comments import YoutubeCommentsError, analyze_video_comments
from .services.youtube_service import YoutubeServiceError

bp = Blueprint("routes", __name__)

_SYNTHESIS_MAX_SOURCES = 4


@bp.get("/")
def index():
    return render_template("index.html")


def _render_result(
    template: str,
    *,
    source_text: str,
    error: str | None,
    source_type: str,
    source_ref: str = "",
    **extra,
):
    """Shared helper for the classic synchronous routes (the no-JavaScript
    fallback -- see services/jobs.py for why the async path exists
    alongside this one)."""
    summary = None
    keywords: list[str] = []
    used_textrank = False
    degraded = False
    doc_id = None
    detail = request.form.get("detail", "standard")

    if source_text and not error:
        try:
            outcome = pipeline.run_summarize_pipeline(current_user.id, source_text, detail, source_type, source_ref)
            summary = outcome["summary"]
            used_textrank = outcome["used_textrank"]
            keywords = outcome["keywords"]
            doc_id = outcome["doc_id"]
            degraded = outcome.get("degraded", False)
        except SummarizerError as exc:
            error = str(exc)

    return render_template(
        template,
        summary=summary,
        keywords=keywords,
        error=error,
        used_textrank=used_textrank,
        degraded=degraded,
        doc_id=doc_id,
        detail=detail,
        source_ref=source_ref,
        **extra,
    )


@bp.route("/text", methods=["GET", "POST"])
@login_required
def text():
    if request.method == "GET":
        return render_template("text.html", detail="standard")

    raw_text = request.form.get("input_text", "")
    error = None if raw_text.strip() else "Please paste some text to summarize."
    return _render_result(
        "text.html",
        source_text=raw_text,
        error=error,
        source_type="text",
        source_ref="Pasted text",
        submitted=True,
    )


@bp.route("/pdf", methods=["GET", "POST"])
@login_required
def pdf():
    if request.method == "GET":
        return render_template("pdf.html", detail="standard")

    uploaded = request.files.get("pdf")
    error = None
    extracted = ""
    source_ref = ""
    if not uploaded or not uploaded.filename:
        error = "Please choose a PDF file."
    else:
        source_ref = uploaded.filename
        try:
            extracted = extract_pdf_text(uploaded.stream)
            if not extracted:
                error = "Couldn't find any text in that PDF (is it scanned images?)."
        except PdfExtractionError as exc:
            error = str(exc)

    return _render_result(
        "pdf.html",
        source_text=extracted,
        error=error,
        source_type="pdf",
        source_ref=source_ref,
        submitted=True,
    )


@bp.route("/audio", methods=["GET", "POST"])
@login_required
def audio():
    """Audio ingestion (services/audio_service.py) -- podcasts, voice
    memos, meeting recordings, or a clip recorded right here in the
    browser (see static/js/recorder.js), understood by Gemini directly,
    no separate transcription step or model to install."""
    if request.method == "GET":
        return render_template("audio.html", detail="standard")

    uploaded = request.files.get("audio")
    detail = request.form.get("detail", "standard")
    error = None
    result = None
    if not uploaded or not uploaded.filename:
        error = "Please choose or record an audio clip."
    else:
        try:
            outcome = pipeline.run_audio_job(current_user.id, uploaded.read(), uploaded.filename, detail)
            result = outcome
        except AudioServiceError as exc:
            error = str(exc)

    return render_template(
        "audio.html",
        summary=result["summary"] if result else None,
        keywords=result["keywords"] if result else [],
        doc_id=result["doc_id"] if result else None,
        source_ref=result["source_ref"] if result else "",
        error=error,
        detail=detail,
        submitted=True,
    )


@bp.route("/image", methods=["GET", "POST"])
@login_required
def image():
    """Image/screenshot understanding (services/image_service.py) --
    infographics, whiteboard photos, receipts, charts, or a photo snapped
    right here with your webcam (see static/js/recorder.js)."""
    if request.method == "GET":
        return render_template("image.html", modes=ANALYSIS_MODES, mode="describe")

    uploaded = request.files.get("image")
    mode = request.form.get("mode", "describe")
    error = None
    result = None
    if not uploaded or not uploaded.filename:
        error = "Please choose or capture an image."
    else:
        try:
            outcome = pipeline.run_image_job(current_user.id, uploaded.read(), uploaded.filename, mode)
            result = outcome
        except ImageServiceError as exc:
            error = str(exc)

    return render_template(
        "image.html",
        modes=ANALYSIS_MODES,
        mode=mode,
        summary=result["summary"] if result else None,
        keywords=result["keywords"] if result else [],
        doc_id=result["doc_id"] if result else None,
        source_ref=result["source_ref"] if result else "",
        error=error,
        submitted=True,
    )


@bp.route("/video", methods=["GET", "POST"])
@login_required
def video():
    """Video ingestion (services/video_service.py) -- a screen capture,
    a clip you upload, or a video recorded right here with your webcam
    and mic together (see static/js/recorder.js). Gemini understands the
    visual track and the audio track natively, in one pass."""
    if request.method == "GET":
        return render_template("video.html", detail="standard")

    uploaded = request.files.get("video")
    detail = request.form.get("detail", "standard")
    error = None
    result = None
    if not uploaded or not uploaded.filename:
        error = "Please choose or record a video clip."
    else:
        try:
            outcome = pipeline.run_video_job(current_user.id, uploaded.read(), uploaded.filename, detail)
            result = outcome
        except VideoServiceError as exc:
            error = str(exc)

    return render_template(
        "video.html",
        summary=result["summary"] if result else None,
        keywords=result["keywords"] if result else [],
        doc_id=result["doc_id"] if result else None,
        source_ref=result["source_ref"] if result else "",
        error=error,
        detail=detail,
        submitted=True,
    )


@bp.route("/youtube", methods=["GET", "POST"])
@login_required
def youtube():
    if request.method == "GET":
        return render_template("youtube.html", detail="standard")

    url = request.form.get("url_youtube", "")
    detail = request.form.get("detail", "standard")
    error = None
    result = None
    if not url.strip():
        error = "Please paste a YouTube video URL."
    else:
        try:
            outcome = pipeline.run_youtube_job(current_user.id, url, detail)
            result = outcome
        except YoutubeServiceError as exc:
            error = str(exc)
        except SummarizerError as exc:
            error = str(exc)

    return render_template(
        "youtube.html",
        summary=result["summary"] if result else None,
        video_id=result["video_id"] if result else None,
        method=result["method"] if result else None,
        keywords=result["keywords"] if result else [],
        doc_id=result["doc_id"] if result else None,
        error=error,
        detail=detail,
        submitted=True,
    )


@bp.route("/article", methods=["GET", "POST"])
@login_required
def article():
    if request.method == "GET":
        return render_template("article.html", detail="standard")

    url = request.form.get("url", "")
    error = None
    extracted = ""
    if not url.strip():
        error = "Please paste an article URL."
    else:
        try:
            extracted = extract_article_text(url)
        except ArticleExtractionError as exc:
            error = str(exc)

    return _render_result(
        "article.html",
        source_text=extracted,
        error=error,
        source_type="article",
        source_ref=url,
        submitted=True,
    )


@bp.route("/history", methods=["GET"])
@login_required
def history():
    query = request.args.get("q", "").strip()
    error = None
    results = []
    if query:
        try:
            results = history_store.search_history(current_user.id, query)
        except history_store.HistoryStoreError as exc:
            error = str(exc)
    else:
        results = history_store.list_recent(current_user.id, limit=30)

    return render_template(
        "history.html", query=query, results=results, error=error, searched=bool(query)
    )


@bp.route("/search", methods=["GET"])
@login_required
def search_page():
    """Real local search engine over your own knowledge base
    (services/search.py) -- instant keyword matching that needs no API
    call, layered with semantic ranking when embeddings are available.
    Not the same thing as the command palette's quick lookup: this is the
    full search experience, with match type and similarity shown."""
    query = request.args.get("q", "").strip()
    results = search_service.search(current_user.id, query, limit=30) if query else []
    return render_template("search.html", query=query, results=results, searched=bool(query))


@bp.route("/analytics", methods=["GET"])
@login_required
def analytics_page():
    """Personal analytics (services/analytics.py) -- real aggregate
    statistics computed directly from your own stored data, no Gemini
    call involved."""
    return render_template("analytics.html", stats=analytics.get_analytics(current_user.id))


@bp.route("/graph", methods=["GET"])
@login_required
def graph_page():
    """Whole-knowledge-base view of the auto-linked graph
    (services/knowledge_graph.py) -- every summarized source as a node,
    every Gemini-classified relationship between two of them as a colored
    edge, rendered as a plain SVG (no charting library needed at this
    scale)."""
    graph = history_store.graph_data(current_user.id)
    svg = knowledge_graph.render_graph_svg(graph)
    return render_template("graph.html", svg=svg, node_count=len(graph["nodes"]), edge_count=len(graph["edges"]))


@bp.get("/api/graph/<int:entry_id>")
@login_required
def api_graph_entry(entry_id):
    """AJAX endpoint: everything auto-linked to one knowledge-base entry,
    for the 'Related' expander on the history page."""
    return jsonify({"links": history_store.get_links_for(current_user.id, entry_id)})


@bp.post("/api/annotations")
@login_required
def api_annotations_add():
    """AJAX endpoint: save a personal highlight + note on a knowledge-base
    entry (services/annotations.py) -- the one feature here that never
    calls Gemini at all."""
    data = request.get_json(silent=True) or {}
    try:
        entry_id = int(data.get("entry_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing entry_id."}), 400
    try:
        annotation_id = annotations.add_annotation(current_user.id, entry_id, data.get("quote", ""), data.get("note", ""))
    except AnnotationError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"id": annotation_id})


@bp.get("/api/annotations/<int:entry_id>")
@login_required
def api_annotations_list(entry_id):
    return jsonify({"annotations": annotations.list_annotations_for(current_user.id, entry_id)})


@bp.post("/api/annotations/<int:annotation_id>/delete")
@login_required
def api_annotations_delete(annotation_id):
    annotations.delete_annotation(annotation_id, current_user.id)
    return jsonify({"ok": True})


@bp.post("/api/share/<int:entry_id>")
@login_required
def api_share_create(entry_id):
    """AJAX endpoint: mint (or reuse) a public read-only share link for
    one knowledge-base entry (services/sharing.py)."""
    try:
        token = sharing.create_share(current_user.id, entry_id)
    except SharingError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"url": url_for("routes.shared_entry", token=token, _external=True)})


@bp.get("/s/<token>")
def shared_entry(token):
    """Public, read-only view of a shared entry -- no login, no nav into
    the rest of the knowledge base, just the one summary someone chose to
    share."""
    entry = sharing.resolve_share(token)
    if entry is None:
        return render_template("shared.html", entry=None), 404
    return render_template("shared.html", entry=entry)


@bp.get("/save")
def bookmarklet_save():
    """One-click save-from-anywhere (the browser bookmarklet built on the
    profile page). A bookmarklet runs on whatever page you're currently
    looking at, not on this app's own origin, so it can't send a login
    cookie -- instead it opens this URL with the page's own address and
    your personal, revocable api_token (see auth.py's /profile) as query
    parameters."""
    token = request.args.get("token", "")
    url = request.args.get("url", "")
    user = User.query.filter_by(api_token=token).first() if token else None

    if user is None:
        return render_template("bookmarklet_result.html", error="That save link isn't valid. Generate a fresh one from your profile page."), 401
    if not url.strip():
        return render_template("bookmarklet_result.html", error="No page URL was given to save.", user=user), 400

    try:
        extracted = extract_article_text(url)
        if not extracted:
            raise ArticleExtractionError("Couldn't find readable article text on that page.")
        outcome = pipeline.run_summarize_pipeline(user.id, extracted, "standard", "article", url)
    except (ArticleExtractionError, SummarizerError) as exc:
        return render_template("bookmarklet_result.html", error=str(exc), user=user, url=url)

    return render_template(
        "bookmarklet_result.html",
        user=user,
        url=url,
        summary=outcome["summary"],
        doc_id=outcome["doc_id"],
    )


@bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard_page():
    """Ops/observability dashboard -- real internal tooling, not an AI
    feature: Gemini resilience-layer health (gemini_client.py), cache hit
    rate (cache.py), background-job status (jobs.py), and watchlist
    health (watchlist.py), all in one place. The kind of page a
    production app has and a demo app usually doesn't."""
    return render_template(
        "dashboard.html",
        gemini_health=gemini_client.get_health_status(),
        cache_stats=cache.get_stats(),
        job_stats=jobs.get_stats(current_user.id),
        watch_health=watchlist.health_stats(current_user.id),
        entry_count=history_store.entry_count(current_user.id),
        annotation_count=annotations.annotation_count(current_user.id),
        card_count=spaced_repetition.total_count(current_user.id),
    )


@bp.get("/api/dashboard")
@login_required
def api_dashboard():
    """Same data as /dashboard, as JSON -- lets the page auto-refresh
    without a full reload."""
    return jsonify(
        {
            "gemini_health": gemini_client.get_health_status(),
            "cache_stats": cache.get_stats(),
            "job_stats": jobs.get_stats(current_user.id),
            "watch_health": watchlist.health_stats(current_user.id),
            "entry_count": history_store.entry_count(current_user.id),
            "annotation_count": annotations.annotation_count(current_user.id),
            "card_count": spaced_repetition.total_count(current_user.id),
        }
    )


@bp.route("/compose", methods=["GET", "POST"])
@login_required
def compose_page():
    """Draft/Compose mode (services/compose.py) -- generate new content
    (email, social post, follow-up, blog intro) from a knowledge-base
    entry or pasted text, with every regeneration saved as a new,
    diffable version rather than overwriting the last one."""
    entry_id = request.values.get("entry_id", type=int)
    draft_type = request.values.get("draft_type", "email")
    error = None
    draft = None
    diff_html = None

    if request.method == "POST":
        source_text = request.form.get("source_text", "").strip()
        if entry_id and not source_text:
            entry = history_store.get_entry_for_user(entry_id, current_user.id)
            source_text = entry["summary"] if entry else ""
        instructions = request.form.get("instructions", "")
        try:
            content = compose.generate_draft(source_text, draft_type, instructions)
            previous = compose.list_versions(current_user.id, entry_id, draft_type)
            saved = compose.save_version(current_user.id, entry_id, draft_type, instructions, content)
            draft = {"content": content, "version": saved["version"]}
            if previous:
                diff_html = compose.word_diff_html(previous[0]["content"], content)
        except ComposeError as exc:
            error = str(exc)

    return render_template(
        "compose.html",
        entry_id=entry_id,
        draft_type=draft_type,
        draft_types=compose.DRAFT_TYPES,
        recent_entries=history_store.list_recent(current_user.id, limit=30),
        versions=compose.list_versions(current_user.id, entry_id, draft_type),
        draft=draft,
        diff_html=diff_html,
        error=error,
    )


@bp.route("/digest", methods=["GET"])
@login_required
def digest_page():
    """Weekly digest (services/digest.py) -- everything summarized plus
    everything the watchlist caught changing in the last 7 days, rolled
    into one 'week in review'. Generated once per calendar week and
    cached from there."""
    return render_template("digest.html", digest=digest.build_weekly_digest(current_user.id))


@bp.route("/duplicates", methods=["GET"])
@login_required
def duplicates_page():
    """Duplicate detection (services/dedup.py) -- near-identical
    knowledge-base entries, grouped by embedding similarity, with a
    one-click merge."""
    return render_template("duplicates.html", groups=dedup.find_duplicate_groups(current_user.id))


@bp.post("/duplicates/merge")
@login_required
def duplicates_merge():
    keep_id = request.form.get("keep_id", type=int)
    duplicate_ids = [int(v) for v in request.form.getlist("duplicate_ids") if v.strip()]
    if keep_id and duplicate_ids:
        dedup.merge_group(current_user.id, keep_id, duplicate_ids)
    return redirect(url_for("routes.duplicates_page"))


@bp.route("/export", methods=["GET"])
@login_required
def export_page():
    """Export center (services/export_center.py) -- your data leaves this
    app in formats real tools already understand: an Anki deck, an
    Obsidian-compatible Markdown vault, a PDF booklet, or a plain JSON
    backup of everything (see /account/export)."""
    return render_template(
        "export.html",
        card_count=spaced_repetition.total_count(current_user.id),
        entry_count=history_store.entry_count(current_user.id),
    )


@bp.get("/export/anki")
@login_required
def export_anki():
    try:
        data = export_center.build_anki_deck(current_user.id)
    except ExportError as exc:
        return render_template("export.html", error=str(exc), card_count=spaced_repetition.total_count(current_user.id), entry_count=history_store.entry_count(current_user.id))
    return Response(
        data, mimetype="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=summarease.apkg"},
    )


@bp.get("/export/obsidian")
@login_required
def export_obsidian():
    try:
        data = export_center.build_obsidian_vault(current_user.id)
    except ExportError as exc:
        return render_template("export.html", error=str(exc), card_count=spaced_repetition.total_count(current_user.id), entry_count=history_store.entry_count(current_user.id))
    return Response(
        data, mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=summarease-vault.zip"},
    )


@bp.get("/export/pdf")
@login_required
def export_pdf():
    try:
        data = export_center.build_knowledge_base_pdf(current_user.id)
    except ExportError as exc:
        return render_template("export.html", error=str(exc), card_count=spaced_repetition.total_count(current_user.id), entry_count=history_store.entry_count(current_user.id))
    return Response(
        data, mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=summarease-knowledge-base.pdf"},
    )


@bp.get("/account/export")
@login_required
def account_export():
    """Full account backup (services/backup.py) -- every row this
    account owns, as one plain JSON file. Not export-of-a-feature like
    the ones above; this is everything, portable to another account or
    another SummarEase instance entirely."""
    data = backup.export_account(current_user.id)
    return Response(
        json.dumps(data, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=summarease-backup.json"},
    )


@bp.post("/account/import")
@login_required
def account_import():
    uploaded = request.files.get("backup_file")
    if not uploaded or not uploaded.filename:
        return render_template("auth/profile.html", error="Please choose a backup JSON file.", stats=_profile_stats(), bookmarklet_url=url_for("routes.bookmarklet_save", _external=True))
    try:
        data = json.loads(uploaded.read().decode("utf-8"))
        counts = backup.import_account(current_user.id, data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return render_template("auth/profile.html", error="Couldn't read that file as a SummarEase backup.", stats=_profile_stats(), bookmarklet_url=url_for("routes.bookmarklet_save", _external=True))
    except BackupError as exc:
        return render_template("auth/profile.html", error=str(exc), stats=_profile_stats(), bookmarklet_url=url_for("routes.bookmarklet_save", _external=True))

    success = f"Imported {counts['entries']} knowledge-base entries, {counts['flashcards']} flashcards, {counts['annotations']} annotations, and {counts['watches']} watches."
    return render_template("auth/profile.html", success=success, stats=_profile_stats(), bookmarklet_url=url_for("routes.bookmarklet_save", _external=True))


def _profile_stats() -> dict:
    from .models import Annotation, Draft, Flashcard, HistoryEntry, Watch

    return {
        "entry_count": HistoryEntry.query.filter_by(user_id=current_user.id).count(),
        "watch_count": Watch.query.filter_by(user_id=current_user.id).count(),
        "flashcard_count": Flashcard.query.filter_by(user_id=current_user.id).count(),
        "annotation_count": Annotation.query.filter_by(user_id=current_user.id).count(),
        "draft_count": Draft.query.filter_by(user_id=current_user.id).count(),
    }


@bp.route("/watchlist", methods=["GET", "POST"])
@login_required
def watchlist_page():
    """Autonomous watchlist -- add a page to watch and a background
    thread (services/watchlist.py) re-checks it on its own schedule,
    writing a Gemini-generated 'what changed' note to the digest feed
    whenever the content actually changes. Unlike every other feature in
    this app, this one keeps working when nobody's looking at the page."""
    error = None
    if request.method == "POST":
        url = request.form.get("url", "")
        label = request.form.get("label", "")
        try:
            interval_minutes = int(request.form.get("interval_minutes", 60))
        except ValueError:
            interval_minutes = 60
        try:
            watchlist.add_watch(current_user.id, url, label, interval_minutes)
        except WatchlistError as exc:
            error = str(exc)
        else:
            return redirect(url_for("routes.watchlist_page"))

    return render_template(
        "watchlist.html",
        watches=watchlist.list_watches(current_user.id),
        digest=watchlist.list_digest(current_user.id, limit=50),
        error=error,
    )


@bp.post("/watchlist/<int:watch_id>/remove")
@login_required
def watchlist_remove(watch_id):
    watchlist.remove_watch(watch_id, current_user.id)
    return redirect(url_for("routes.watchlist_page"))


@bp.post("/watchlist/<int:watch_id>/toggle")
@login_required
def watchlist_toggle(watch_id):
    watch = watchlist.get_watch(watch_id, current_user.id)
    if watch is not None:
        watchlist.toggle_watch(watch_id, current_user.id, not watch["active"])
    return redirect(url_for("routes.watchlist_page"))


@bp.post("/watchlist/mark_read")
@login_required
def watchlist_mark_read():
    watchlist.mark_all_read(current_user.id)
    return redirect(url_for("routes.watchlist_page"))


@bp.post("/api/watchlist/<int:watch_id>/check")
@login_required
def api_watchlist_check(watch_id):
    """AJAX endpoint: force an immediate check of one watch (bypassing
    its interval) so a demo doesn't have to wait for the schedule."""
    try:
        result = watchlist.check_now(watch_id, current_user.id)
    except WatchlistError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@bp.get("/api/watchlist/unread_count")
@login_required
def api_watchlist_unread_count():
    return jsonify({"unread_count": watchlist.unread_count(current_user.id)})


@bp.route("/review", methods=["GET"])
@login_required
def review_page():
    """Spaced-repetition review queue (services/spaced_repetition.py,
    real SM-2 scheduling). Cards are generated on demand from any result
    via 'Generate flashcards' and reviewed here one at a time."""
    return render_template(
        "review.html",
        due=spaced_repetition.due_flashcards(current_user.id, limit=20),
        due_count=spaced_repetition.due_count(current_user.id),
        total_count=spaced_repetition.total_count(current_user.id),
    )


@bp.post("/api/flashcards/generate")
@login_required
def api_flashcards_generate():
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "")
    source_ref = data.get("source_ref", "this document")
    text_value = rag_get_full_text(current_user.id, doc_id)
    if not text_value:
        return jsonify(
            {"error": "This document is no longer available -- try summarizing it again."}
        ), 400
    try:
        count = spaced_repetition.generate_and_save(current_user.id, text_value, source_ref)
    except SpacedRepetitionError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"count": count})


@bp.post("/api/flashcards/grade")
@login_required
def api_flashcards_grade():
    data = request.get_json(silent=True) or {}
    card_id = data.get("card_id")
    quality = data.get("quality")
    try:
        result = spaced_repetition.grade_flashcard(card_id, current_user.id, quality)
    except (SpacedRepetitionError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@bp.get("/api/flashcards/due_count")
@login_required
def api_flashcards_due_count():
    return jsonify({"due_count": spaced_repetition.due_count(current_user.id)})


def _parse_multi_source_form(max_sources: int) -> list[dict]:
    """Read the 'Source N' rows (kind/label/text/url/pdf) shared by
    Briefing mode and Compare mode out of the submitted form."""
    rows = []
    for i in range(1, max_sources + 1):
        kind = request.form.get(f"kind_{i}", "skip")
        label = request.form.get(f"label_{i}", "").strip() or f"Source {i}"
        if kind == "skip":
            continue
        uploaded = request.files.get(f"file_{i}") if kind == "pdf" else None
        pdf_bytes = uploaded.read() if (uploaded and uploaded.filename) else None
        rows.append(
            {
                "kind": kind,
                "label": label,
                "text_value": request.form.get(f"text_value_{i}", "").strip(),
                "url_value": request.form.get(f"url_value_{i}", "").strip(),
                "pdf_bytes": pdf_bytes,
                "pdf_name": uploaded.filename if uploaded else None,
            }
        )
    return rows


def _extract_multi_sources(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Extract text for each source row concurrently (independent I/O --
    URL fetches, PDF parses -- run in parallel rather than one after
    another). Returns (sources, warnings)."""
    from concurrent.futures import ThreadPoolExecutor

    def _extract(row):
        if row["kind"] == "text":
            return row["label"], row["text_value"], None
        if row["kind"] == "url" and row["url_value"]:
            try:
                return row["label"], extract_article_text(row["url_value"]), None
            except ArticleExtractionError as exc:
                return row["label"], "", str(exc)
        if row["kind"] == "pdf" and row["pdf_bytes"]:
            import io

            try:
                return row["label"], extract_pdf_text(io.BytesIO(row["pdf_bytes"])), None
            except PdfExtractionError as exc:
                return row["label"], "", str(exc)
        return row["label"], "", "no usable content found, skipped."

    sources = []
    warnings = []
    if rows:
        with ThreadPoolExecutor(max_workers=len(rows)) as executor:
            for label, text_value, warning in executor.map(_extract, rows):
                if text_value:
                    sources.append({"label": label, "text": text_value})
                elif warning:
                    warnings.append(f"{label}: {warning}")
    return sources, warnings


@bp.route("/compare", methods=["GET", "POST"])
@login_required
def compare():
    """Structured multi-document comparison (services/comparator.py) --
    the same multi-source form as Briefing mode, but the output is a
    Gemini-schema-constrained comparison table instead of synthesized
    prose, because 'compare these' and 'summarize these together' are
    genuinely different asks."""
    if request.method == "GET":
        return render_template("compare.html", max_sources=_SYNTHESIS_MAX_SOURCES)

    rows = _parse_multi_source_form(_SYNTHESIS_MAX_SOURCES)
    sources, warnings = _extract_multi_sources(rows)

    error = None
    result = None
    if len(sources) < 2:
        error = (
            "Add at least two sources with usable content (text, a working "
            "article URL, or a PDF with extractable text)."
        )
    else:
        try:
            result = comparator.compare_sources(sources)
        except comparator.ComparatorError as exc:
            error = str(exc)

    return render_template(
        "compare.html",
        max_sources=_SYNTHESIS_MAX_SOURCES,
        result=result,
        error=error,
        warnings=warnings,
        submitted=True,
    )


@bp.route("/briefing", methods=["GET", "POST"])
@login_required
def briefing():
    if request.method == "GET":
        return render_template("briefing.html", max_sources=_SYNTHESIS_MAX_SOURCES)

    rows = _parse_multi_source_form(_SYNTHESIS_MAX_SOURCES)
    sources, warnings = _extract_multi_sources(rows)

    error = None
    result = None
    if len(sources) < 2:
        error = (
            "Add at least two sources with usable content (text, a working "
            "article URL, or a PDF with extractable text)."
        )
    else:
        try:
            result = synthesize_sources(sources)
        except SynthesisError as exc:
            error = str(exc)

    doc_id = None
    if result:
        combined_text = "\n\n".join(s["text"] for s in sources)[:20000]
        try:
            entry_id = history_store.save_entry(
                current_user.id, "briefing", ", ".join(s["label"] for s in sources), result["briefing"], combined_text
            )
            doc_id = str(entry_id)
            knowledge_graph.auto_link_entry_async(current_user.id, entry_id)
            try:
                rag.index_entry(entry_id, combined_text)
            except Exception:
                pass
        except Exception:
            doc_id = None

    return render_template(
        "briefing.html",
        max_sources=_SYNTHESIS_MAX_SOURCES,
        result=result,
        error=error,
        warnings=warnings,
        doc_id=doc_id,
        submitted=True,
    )


# ---------------------------------------------------------------------------
# Async job endpoints (live-progress path; see services/jobs.py)
# ---------------------------------------------------------------------------


@bp.post("/api/jobs/start")
@login_required
def api_jobs_start():
    kind = request.form.get("kind", "")
    detail = request.form.get("detail", "standard")
    uid = current_user.id

    if kind == "text":
        text_value = request.form.get("input_text", "")
        if not text_value.strip():
            return jsonify({"error": "Please paste some text to summarize."}), 400
        job_id = jobs.start_job(uid, pipeline.run_text_job, uid, text_value, detail, kind="text")

    elif kind == "article":
        url = request.form.get("url", "")
        if not url.strip():
            return jsonify({"error": "Please paste an article URL."}), 400
        job_id = jobs.start_job(uid, pipeline.run_article_job, uid, url, detail, kind="article")

    elif kind == "youtube":
        url = request.form.get("url_youtube", "")
        if not url.strip():
            return jsonify({"error": "Please paste a YouTube video URL."}), 400
        job_id = jobs.start_job(uid, pipeline.run_youtube_job, uid, url, detail, kind="youtube")

    elif kind == "pdf":
        uploaded = request.files.get("pdf")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Please choose a PDF file."}), 400
        job_id = jobs.start_job(uid, pipeline.run_pdf_job, uid, uploaded.read(), uploaded.filename, detail, kind="pdf")

    elif kind == "audio":
        uploaded = request.files.get("audio")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Please choose or record an audio clip."}), 400
        job_id = jobs.start_job(uid, pipeline.run_audio_job, uid, uploaded.read(), uploaded.filename, detail, kind="audio")

    elif kind == "video":
        uploaded = request.files.get("video")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Please choose or record a video clip."}), 400
        job_id = jobs.start_job(uid, pipeline.run_video_job, uid, uploaded.read(), uploaded.filename, detail, kind="video")

    elif kind == "image":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Please choose or capture an image."}), 400
        mode = request.form.get("mode", "describe")
        job_id = jobs.start_job(uid, pipeline.run_image_job, uid, uploaded.read(), uploaded.filename, mode, kind="image")

    else:
        return jsonify({"error": "Unknown job kind."}), 400

    return jsonify({"job_id": job_id})


@bp.get("/api/jobs/<job_id>")
@login_required
def api_jobs_status(job_id):
    job = jobs.get_job(job_id, current_user.id)
    if job is None:
        return jsonify({"error": "Job not found (it may have expired)."}), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# Other AJAX endpoints
# ---------------------------------------------------------------------------


@bp.post("/api/explain")
@login_required
def api_explain():
    """AJAX endpoint: short definition for one extracted keyword."""
    data = request.get_json(silent=True) or {}
    term = data.get("term", "")
    context = data.get("context", "")
    try:
        return jsonify({"term": term, "explanation": explain_term(term, context)})
    except SummarizerError as exc:
        return jsonify({"term": term, "error": str(exc)}), 502


@bp.post("/api/ask")
@login_required
def api_ask():
    """AJAX endpoint: retrieval-augmented Q&A grounded in one indexed
    document (see services/rag.py)."""
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "")
    question = data.get("question", "")
    try:
        return jsonify(rag_ask(current_user.id, doc_id, question))
    except RagError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/api/credibility")
@login_required
def api_credibility():
    """AJAX endpoint: Credibility & Framing Lens for one indexed document."""
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "")
    text_value = rag_get_full_text(current_user.id, doc_id)
    if not text_value:
        return jsonify(
            {"error": "This document is no longer available -- try summarizing it again."}
        ), 400
    try:
        return jsonify(analyze_credibility(text_value))
    except CredibilityError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/api/faithfulness")
@login_required
def api_faithfulness():
    """AJAX endpoint: hallucination/faithfulness self-check -- does the
    summary say anything the source doesn't actually support?"""
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "")
    summary = data.get("summary", "")
    source_text = rag_get_full_text(current_user.id, doc_id)
    if not source_text:
        return jsonify(
            {"error": "This document is no longer available -- try summarizing it again."}
        ), 400
    try:
        return jsonify(check_faithfulness(source_text, summary))
    except SummarizerError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/api/perspectives")
@login_required
def api_perspectives():
    """AJAX endpoint: steelmanned multi-perspective view of one indexed
    document (see services/perspectives.py)."""
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id", "")
    text_value = rag_get_full_text(current_user.id, doc_id)
    if not text_value:
        return jsonify(
            {"error": "This document is no longer available -- try summarizing it again."}
        ), 400
    try:
        return jsonify(perspectives.generate_perspectives(text_value))
    except PerspectivesError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.get("/api/topics")
@login_required
def api_topics():
    """AJAX endpoint: k-means topic clusters over the knowledge base,
    each labeled by Gemini (see services/topics.py)."""
    try:
        return jsonify({"clusters": get_topic_clusters(current_user.id)})
    except TopicsError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.get("/api/search")
@login_required
def api_search():
    """AJAX endpoint backing the command palette (Cmd/Ctrl+K) -- plain,
    instant keyword search over the knowledge base, no API call. Page
    navigation results are matched client-side (see app.js); this only
    covers your actual data."""
    query = request.args.get("q", "")
    return jsonify({"results": history_store.keyword_search(current_user.id, query, limit=8)})


@bp.get("/api/stats")
@login_required
def api_stats():
    return jsonify({"entry_count": history_store.entry_count(current_user.id)})


@bp.post("/api/youtube/sentiment")
@login_required
def api_youtube_sentiment():
    """AJAX endpoint: sentiment of a video's top comments, via the YouTube
    Data API + a lightweight lexicon scorer (see youtube_comments.py)."""
    data = request.get_json(silent=True) or {}
    video_id = data.get("video_id", "")
    if not video_id:
        return jsonify({"error": "Missing video_id."}), 400
    try:
        return jsonify(analyze_video_comments(video_id))
    except YoutubeCommentsError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/api/speak")
@login_required
def api_speak():
    """AJAX endpoint: synthesize `text` to mp3, return its static URL."""
    from flask import current_app

    data = request.get_json(silent=True) or {}
    text_to_speak = data.get("text", "")
    try:
        filename = synthesize_speech(text_to_speak, current_app.static_folder)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"audio_url": url_for("static", filename=f"audio/{filename}")})

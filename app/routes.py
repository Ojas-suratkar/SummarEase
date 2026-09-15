from flask import Blueprint, jsonify, render_template, request

from .services.article_extractor import ArticleExtractionError, extract_article_text
from .services.keywords import extract_keywords
from .services.pdf_extractor import PdfExtractionError, extract_text as extract_pdf_text
from .services.summarizer import SummarizerError, explain_term, summarize_text
from .services.tts_service import synthesize
from .services.youtube_service import YoutubeServiceError, summarize_video

bp = Blueprint("routes", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


def _render_result(template: str, *, source_text: str, error: str | None, **extra):
    """Shared helper: summarize `source_text` (if any) and render a page."""
    summary = None
    keywords: list[str] = []

    if source_text and not error:
        try:
            summary = summarize_text(source_text)
            keywords = extract_keywords(source_text)
        except SummarizerError as exc:
            error = str(exc)

    return render_template(
        template,
        summary=summary,
        keywords=keywords,
        error=error,
        **extra,
    )


@bp.route("/text", methods=["GET", "POST"])
def text():
    if request.method == "GET":
        return render_template("text.html")

    raw_text = request.form.get("input_text", "")
    error = None if raw_text.strip() else "Please paste some text to summarize."
    return _render_result("text.html", source_text=raw_text, error=error, submitted=True)


@bp.route("/pdf", methods=["GET", "POST"])
def pdf():
    if request.method == "GET":
        return render_template("pdf.html")

    uploaded = request.files.get("pdf")
    error = None
    extracted = ""
    if not uploaded or not uploaded.filename:
        error = "Please choose a PDF file."
    else:
        try:
            extracted = extract_pdf_text(uploaded.stream)
            if not extracted:
                error = "Couldn't find any text in that PDF (is it scanned images?)."
        except PdfExtractionError as exc:
            error = str(exc)

    return _render_result("pdf.html", source_text=extracted, error=error, submitted=True)


@bp.route("/youtube", methods=["GET", "POST"])
def youtube():
    if request.method == "GET":
        return render_template("youtube.html")

    url = request.form.get("url_youtube", "")
    error = None
    result = None
    if not url.strip():
        error = "Please paste a YouTube video URL."
    else:
        try:
            result = summarize_video(url)
        except YoutubeServiceError as exc:
            error = str(exc)
        except SummarizerError as exc:
            error = str(exc)

    summary = result["summary"] if result else None
    keywords = extract_keywords(summary) if summary else []
    return render_template(
        "youtube.html",
        summary=summary,
        keywords=keywords,
        error=error,
        submitted=True,
    )


@bp.route("/article", methods=["GET", "POST"])
def article():
    if request.method == "GET":
        return render_template("article.html")

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

    return _render_result("article.html", source_text=extracted, error=error, submitted=True)


@bp.post("/api/explain")
def api_explain():
    """AJAX endpoint: short definition for one extracted keyword."""
    data = request.get_json(silent=True) or {}
    term = data.get("term", "")
    context = data.get("context", "")
    try:
        return jsonify({"term": term, "explanation": explain_term(term, context)})
    except SummarizerError as exc:
        return jsonify({"term": term, "error": str(exc)}), 502


@bp.post("/api/speak")
def api_speak():
    """AJAX endpoint: synthesize `text` to mp3, return its static URL."""
    from flask import current_app, url_for

    data = request.get_json(silent=True) or {}
    text_to_speak = data.get("text", "")
    try:
        filename = synthesize(text_to_speak, current_app.static_folder)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"audio_url": url_for("static", filename=f"audio/{filename}")})

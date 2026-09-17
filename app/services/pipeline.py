"""
Shared summarization pipeline. Both the classic synchronous routes
(routes.py's `_render_result` -- the reliable full-page-reload fallback
that works with JavaScript disabled) and the async `/api/jobs/*`
endpoints (the live-progress path the frontend uses when JavaScript is
available) call the same functions here, so the two paths can never
silently drift apart. `report` is a callable each step calls with a
short human-readable status message; the synchronous path passes a
no-op, the async path passes something that appends to a job's progress
log (see jobs.py).

Every function here takes `user_id` and threads it through to
history_store/rag so the result is saved under, and only ever readable
by, whoever asked for it.
"""
from __future__ import annotations

import io

from . import history_store, knowledge_graph, rag, source_store
from .article_extractor import ArticleExtractionError, extract_article_text
from .audio_service import summarize_audio
from .image_service import analyze_image
from .keywords import extract_keywords
from .pdf_extractor import PdfExtractionError, extract_text as extract_pdf_text
from .summarizer import summarize_adaptive
from .video_service import summarize_video_file
from .youtube_service import summarize_video


def _keep_source(user_id: int, entry_id: int, *, kind: str, file_bytes: bytes | None = None, filename: str = "", mime_type: str = "", url: str = "") -> None:
    """Store the original alongside the summary.

    Best-effort on purpose: if the disk is full or the path is
    unwritable, the user still gets the summary they asked for. The
    failure costs replay, not the request.
    """
    try:
        if file_bytes:
            source_store.store_bytes(
                user_id, file_bytes, kind=kind, filename=filename,
                mime_type=mime_type, entry_id=entry_id, original_url=url,
            )
        elif url:
            source_store.store_reference(user_id, kind=kind, original_url=url, entry_id=entry_id)
    except Exception:
        pass


def _save_and_index(user_id: int, source_type: str, source_ref: str, summary: str, source_text: str) -> str | None:
    """Save to the knowledge base, then index the same row for RAG --
    `doc_id` is just the resulting entry's id (see rag.py's module
    docstring for why unifying these two used to be the bug). Both steps
    are best-effort: a knowledge-base or indexing failure never breaks
    the summarization result the user is actually waiting on."""
    try:
        entry_id = history_store.save_entry(user_id, source_type, source_ref, summary, source_text)
    except Exception:
        return None

    knowledge_graph.auto_link_entry_async(user_id, entry_id)

    try:
        rag.index_entry(entry_id, source_text or summary)
    except Exception:
        pass  # still saved -- Q&A/etc. will lazily retry indexing on first use

    return str(entry_id)


def run_summarize_pipeline(user_id: int, text: str, detail: str, source_type: str, source_ref: str, report=lambda msg: None) -> dict:
    word_count = len(text.split())
    if word_count > 600:
        report(f"Long source ({word_count} words) -- ranking sentences with TextRank...")
    report("Summarizing with Gemini...")
    adaptive = summarize_adaptive(text, detail=detail)

    if not adaptive["summary"]:
        return {
            "summary": "", "used_textrank": False, "keywords": [], "doc_id": None,
            "source_ref": source_ref, "degraded": adaptive.get("degraded", False),
        }

    report("Extracting keywords...")
    keywords = extract_keywords(text)

    report("Saving to knowledge base and indexing for Q&A...")
    doc_id = _save_and_index(user_id, source_type, source_ref, adaptive["summary"], text)

    report("Done.")
    return {
        "summary": adaptive["summary"],
        "used_textrank": adaptive["used_textrank"],
        "keywords": keywords,
        "doc_id": doc_id,
        "source_ref": source_ref,
        "degraded": adaptive.get("degraded", False),
    }


def run_text_job(user_id: int, text: str, detail: str, *, report=lambda msg: None) -> dict:
    return run_summarize_pipeline(user_id, text, detail, "text", "Pasted text", report)


def run_article_job(user_id: int, url: str, detail: str, *, report=lambda msg: None) -> dict:
    report("Fetching the article...")
    extracted = extract_article_text(url)  # raises ArticleExtractionError
    result = run_summarize_pipeline(user_id, extracted, detail, "article", url, report)
    if result.get("doc_id"):
        _keep_source(user_id, int(result["doc_id"]), kind="url", url=url)
    return result


def run_pdf_job(user_id: int, file_bytes: bytes, filename: str, detail: str, *, report=lambda msg: None) -> dict:
    report("Extracting text from the PDF...")
    extracted = extract_pdf_text(io.BytesIO(file_bytes))
    if not extracted:
        raise PdfExtractionError("Couldn't find any text in that PDF (is it scanned images?).")
    result = run_summarize_pipeline(user_id, extracted, detail, "pdf", filename, report)
    if result.get("doc_id"):
        _keep_source(user_id, int(result["doc_id"]), kind="pdf", file_bytes=file_bytes,
                     filename=filename, mime_type="application/pdf")
    return result


def run_audio_job(user_id: int, file_bytes: bytes, filename: str, detail: str, *, report=lambda msg: None) -> dict:
    result = summarize_audio(file_bytes, filename, detail=detail, report=report)

    keywords = extract_keywords(result["summary"]) if result["summary"] else []
    doc_id = None
    if result["summary"]:
        report("Saving to knowledge base and indexing for Q&A...")
        doc_id = _save_and_index(user_id, "audio", filename, result["summary"], result["summary"])
        if doc_id:
            _keep_source(user_id, int(doc_id), kind="audio", file_bytes=file_bytes, filename=filename)

    report("Done.")
    return {
        "summary": result["summary"],
        "keywords": keywords,
        "doc_id": doc_id,
        "used_textrank": False,
        "source_ref": filename,
        "degraded": False,
    }


def run_image_job(user_id: int, file_bytes: bytes, filename: str, mode: str, *, report=lambda msg: None) -> dict:
    report("Analyzing image...")
    result = analyze_image(file_bytes, filename, mode=mode)

    keywords = extract_keywords(result["result"]) if result["result"] else []
    doc_id = None
    if result["result"]:
        report("Saving to knowledge base and indexing for Q&A...")
        doc_id = _save_and_index(user_id, "image", filename, result["result"], result["result"])
        if doc_id:
            _keep_source(user_id, int(doc_id), kind="image", file_bytes=file_bytes, filename=filename)

    report("Done.")
    return {
        "summary": result["result"],
        "keywords": keywords,
        "doc_id": doc_id,
        "used_textrank": False,
        "source_ref": filename,
        "degraded": False,
        "mode": mode,
    }


def run_video_job(user_id: int, file_bytes: bytes, filename: str, detail: str, *, report=lambda msg: None) -> dict:
    """A recorded/uploaded video (services/video_service.py) -- handed to
    Gemini directly the same way a YouTube URL is, since Gemini
    understands video natively (visual + audio), not just its audio
    track."""
    result = summarize_video_file(file_bytes, filename, detail=detail, report=report)

    keywords = extract_keywords(result["summary"]) if result["summary"] else []
    doc_id = None
    if result["summary"]:
        report("Saving to knowledge base and indexing for Q&A...")
        doc_id = _save_and_index(user_id, "video", filename, result["summary"], result["summary"])
        if doc_id:
            _keep_source(user_id, int(doc_id), kind="video", file_bytes=file_bytes, filename=filename)

    report("Done.")
    return {
        "summary": result["summary"],
        "keywords": keywords,
        "doc_id": doc_id,
        "used_textrank": False,
        "source_ref": filename,
        "degraded": False,
    }


def run_youtube_job(user_id: int, url: str, detail: str, *, report=lambda msg: None) -> dict:
    result = summarize_video(url, detail=detail, report=report)

    keywords = extract_keywords(result["summary"]) if result["summary"] else []
    doc_id = None
    if result["summary"]:
        report("Saving to knowledge base and indexing for Q&A...")
        doc_id = _save_and_index(user_id, "youtube", url, result["summary"], result["full_text"])
        if doc_id:
            _keep_source(user_id, int(doc_id), kind="url", url=url)

    report("Done.")
    return {
        "summary": result["summary"],
        "video_id": result["video_id"],
        "method": result["method"],
        "keywords": keywords,
        "doc_id": doc_id,
        "used_textrank": False,
        "source_ref": f"YouTube video {result['video_id']}" if result["video_id"] else url,
    }

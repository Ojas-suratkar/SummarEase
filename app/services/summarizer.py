"""
Thin wrapper around Google's Gemini API for text summarization.

Design notes
------------
The API key is read from the environment (GEMINI_API_KEY) rather than
hard-coded; see gemini_client.py for the shared client singleton, the
transient-error retry wrapper (`call_with_retry`) used by every Gemini
call in this module, and MODEL_NAME.

`summarize_adaptive` is the hybrid extractive+abstractive engine: for
long documents it runs TextRank (textrank.py, a graph-centrality
algorithm -- no API call) to pick out the most salient sentences first,
then sends only those to Gemini for an abstractive pass. This keeps
token usage down and summary quality more consistent on long inputs than
either truncating or dumping the whole document in.

`summarize_text` is cached (cache.py) by a hash of (model, target
sentence count, exact text) -- summarizing the same input twice, which
happens constantly while testing/demoing, returns instantly on the
second call and costs no API quota.

Offline fallback
-----------------
If Gemini is genuinely unavailable (every model in the fallback chain
failed -- see gemini_client.GeminiUnavailableError), `summarize_text` and
`summarize_adaptive` do not just fail: they fall back to a pure local
TextRank extractive summary (no API call at all) so the user still gets
something useful instead of an error page. `summarize_adaptive` reports
this via `degraded: True` so the UI can say so plainly.
"""
from __future__ import annotations

from .cache import cached_call
from .gemini_client import (
    MODEL_NAME,
    GeminiNotConfiguredError,
    GeminiUnavailableError,
    generate_content_resilient,
)
from .textrank import extractive_summary, split_sentences

# Three detail levels exposed in the UI. `ratio` controls how much of a
# long document TextRank keeps before the abstractive pass; `sentences`
# is the target length of the final summary.
DETAIL_LEVELS = {
    "brief": {"ratio": 0.25, "sentences": 3},
    "standard": {"ratio": 0.4, "sentences": 5},
    "detailed": {"ratio": 0.6, "sentences": 9},
}
_TEXTRANK_THRESHOLD_WORDS = 600


class SummarizerError(RuntimeError):
    """Raised when the summarizer cannot produce a result."""


def _offline_extractive_summary(text: str, sentences: int) -> str:
    """A pure local fallback with no API call at all -- TextRank's
    graph-centrality ranking, same algorithm used to pre-filter long
    documents before the abstractive pass, just used here as the whole
    summary rather than a pre-filter. Not as fluent as Gemini's abstractive
    summary, but it's real, on-topic, and always available."""
    ranked = extractive_summary(text, ratio=1.0, min_sentences=sentences)
    if not ranked:
        ranked = split_sentences(text)
    return " ".join(ranked[:sentences]).strip()


def _summarize_text_status(text: str, sentences: int) -> tuple[str, bool]:
    """Core implementation shared by summarize_text and summarize_adaptive.
    Returns (summary, degraded) -- degraded is True when Gemini was
    unavailable and the offline TextRank fallback was used instead."""

    def _compute() -> str:
        prompt = (
            f"Summarize the following text in about {sentences} sentences. "
            "Keep the summary factual, neutral, and self-contained. "
            "Do not add any preamble like 'Here is a summary' -- just return "
            "the summary itself.\n\n"
            f"TEXT:\n{text}"
        )
        response = generate_content_resilient(prompt)
        return (response.text or "").strip()

    try:
        result, _cache_hit = cached_call("summarize_text", [MODEL_NAME, str(sentences), text], _compute)
        return result, False
    except GeminiNotConfiguredError as exc:
        raise SummarizerError(str(exc)) from exc
    except GeminiUnavailableError:
        return _offline_extractive_summary(text, sentences), True
    except Exception as exc:  # pragma: no cover - network/API errors
        raise SummarizerError(f"Summarization failed: {exc}") from exc


def summarize_text(text: str, sentences: int = 5) -> str:
    """Summarize arbitrary text down to roughly `sentences` sentences.
    Falls back to a local extractive summary (no API call) if Gemini is
    genuinely unavailable -- see module docstring."""
    text = (text or "").strip()
    if not text:
        return ""
    summary, _degraded = _summarize_text_status(text, sentences)
    return summary


def summarize_adaptive(text: str, detail: str = "standard") -> dict:
    """Summarize `text` at one of DETAIL_LEVELS ('brief'/'standard'/
    'detailed'). Long documents are pre-filtered with TextRank before the
    abstractive Gemini pass; short ones go straight to Gemini. `degraded`
    is True when Gemini was unavailable and the offline TextRank fallback
    was used instead -- the UI surfaces this rather than hiding it.
    """
    text = (text or "").strip()
    if not text:
        return {"summary": "", "used_textrank": False, "detail": detail, "degraded": False}

    settings = DETAIL_LEVELS.get(detail, DETAIL_LEVELS["standard"])
    word_count = len(text.split())
    used_textrank = word_count > _TEXTRANK_THRESHOLD_WORDS

    source = text
    if used_textrank:
        salient = extractive_summary(text, ratio=settings["ratio"])
        source = " ".join(salient) if salient else text

    summary, degraded = _summarize_text_status(source, settings["sentences"])
    return {"summary": summary, "used_textrank": used_textrank, "detail": detail, "degraded": degraded}


def explain_term(term: str, context: str = "") -> str:
    """Return a short, plain-language explanation of a keyword/term.

    This replaces the original project's mislabeled "scrape_wikipedia"
    helper, which didn't actually scrape Wikipedia -- it asked an LLM for
    a definition. This version is honest about what it does.
    """
    term = (term or "").strip()
    if not term:
        return ""

    prompt = (
        f"In 2-3 sentences, explain what '{term}' means"
        + (f", as used in this context: {context[:500]}" if context else "")
        + ". Keep it simple and avoid jargon."
    )
    try:
        response = generate_content_resilient(prompt)
        return (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise SummarizerError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise SummarizerError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise SummarizerError(f"Definition lookup failed: {exc}") from exc


def summarize_youtube_url(youtube_url: str, sentences: int = 6) -> str:
    """Summarize a YouTube video straight from its URL.

    Gemini processes the video's actual audio/visual stream when given a
    public YouTube URL -- it does not need a captions/transcript track at
    all, so this works on videos where `youtube_transcript_api` finds
    nothing (auto-captions disabled, no captions ever uploaded, a
    transient block, etc). Used as the fallback in
    youtube_service.summarize_video.
    """
    from google.genai import types

    prompt = (
        f"Watch this video and summarize it in about {sentences} sentences. "
        "Keep the summary factual, neutral, and self-contained. "
        "Do not add any preamble like 'Here is a summary' -- just return "
        "the summary itself."
    )
    try:
        response = generate_content_resilient(
            types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=youtube_url)),
                    types.Part(text=prompt),
                ]
            )
        )
        return (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise SummarizerError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise SummarizerError(
            f"{exc} This video has no captions, so there's no offline fallback -- please try again shortly."
        ) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise SummarizerError(f"Video summarization failed: {exc}") from exc


def check_faithfulness(source_text: str, summary: str) -> dict:
    """Self-consistency check: ask Gemini whether `summary` is fully
    supported by `source_text`, and list any sentence that isn't (a
    lightweight hallucination/faithfulness check -- the same category of
    technique used in LLM evaluation pipelines, run here as an optional
    second pass a user can trigger on demand).
    """
    source_text = (source_text or "").strip()
    summary = (summary or "").strip()
    if not source_text or not summary:
        return {"faithful": True, "unsupported_claims": []}

    prompt = (
        "You will check whether a SUMMARY is fully supported by a SOURCE "
        "text. List each sentence or claim in the summary that is NOT "
        "clearly supported by the source (a potential inaccuracy or "
        "fabrication), one per line, each prefixed with '- '. Be strict: "
        "only flag things the source doesn't actually say, not minor "
        "phrasing differences. If every claim in the summary is "
        "supported, reply with exactly this and nothing else: "
        "All claims supported.\n\n"
        f"SOURCE:\n{source_text[:6000]}\n\nSUMMARY:\n{summary}"
    )
    try:
        response = generate_content_resilient(prompt)
        raw = (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise SummarizerError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise SummarizerError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise SummarizerError(f"Faithfulness check failed: {exc}") from exc

    if raw.lower().startswith("all claims supported"):
        return {"faithful": True, "unsupported_claims": []}

    unsupported = [line.strip("-* ").strip() for line in raw.splitlines() if line.strip()]
    return {"faithful": len(unsupported) == 0, "unsupported_claims": unsupported}

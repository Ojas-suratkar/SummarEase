"""
Thin wrapper around Google's Gemini API for text summarization.

Design notes
------------
The API key is read from the environment (GEMINI_API_KEY) rather than
hard-coded, and the client is configured lazily on first use so the app
can still start (and its non-LLM features still work) even if no key has
been set yet -- it will just raise a clear error the moment an LLM call
is actually attempted.
"""
from __future__ import annotations

import os
import threading

import google.generativeai as genai

_configure_lock = threading.Lock()
_configured = False


class SummarizerError(RuntimeError):
    """Raised when the summarizer cannot produce a result."""


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SummarizerError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "add your own key from https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        _configured = True


def _model(name: str = "gemini-1.5-flash"):
    _ensure_configured()
    return genai.GenerativeModel(name)


def summarize_text(text: str, sentences: int = 5) -> str:
    """Summarize arbitrary text down to roughly `sentences` sentences."""
    text = (text or "").strip()
    if not text:
        return ""

    prompt = (
        f"Summarize the following text in about {sentences} sentences. "
        "Keep the summary factual, neutral, and self-contained. "
        "Do not add any preamble like 'Here is a summary' -- just return "
        "the summary itself.\n\n"
        f"TEXT:\n{text}"
    )
    try:
        response = _model().generate_content(prompt)
        return (response.text or "").strip()
    except Exception as exc:  # pragma: no cover - network/API errors
        raise SummarizerError(f"Summarization failed: {exc}") from exc


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
        response = _model().generate_content(prompt)
        return (response.text or "").strip()
    except Exception as exc:  # pragma: no cover - network/API errors
        raise SummarizerError(f"Definition lookup failed: {exc}") from exc

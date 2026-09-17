"""
Multi-source synthesis ("Briefing Mode"): combine several sources (any
mix of pasted text, article URLs, and PDFs) into one synthesized
briefing that attributes claims to their source and flags where sources
disagree -- rather than just showing several separate summaries side by
side and leaving the reader to cross-reference them by hand.
"""
from __future__ import annotations

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient

_MAX_SOURCE_CHARS = 6000


class SynthesisError(RuntimeError):
    pass


def synthesize_sources(sources: list[dict], sentences: int = 10) -> dict:
    """`sources` is a list of {"label": str, "text": str} with at least
    two usable entries. Returns a dict with the combined briefing text."""
    usable = [s for s in sources if (s.get("text") or "").strip()]
    if len(usable) < 2:
        raise SynthesisError("Add at least two sources with usable content to synthesize.")

    numbered = "\n\n".join(
        f"SOURCE {i + 1} ({s['label']}):\n{s['text'][:_MAX_SOURCE_CHARS]}"
        for i, s in enumerate(usable)
    )

    prompt = (
        f"You are given {len(usable)} sources on a related topic. Write a "
        f"single synthesized briefing in about {sentences} sentences that "
        "combines what they say, without repeating the same point twice. "
        "When you state something, note which source(s) it came from in "
        "parentheses, e.g. '(Source 1, 3)'. After the briefing, add a "
        "section titled 'Where sources disagree:' listing any factual "
        "contradictions between the sources -- if there are none, write "
        "'No direct contradictions found.'\n\n"
        f"{numbered}"
    )

    try:
        response = generate_content_resilient(prompt)
        briefing = (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise SynthesisError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise SynthesisError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise SynthesisError(f"Synthesis failed: {exc}") from exc

    return {
        "briefing": briefing,
        "source_count": len(usable),
        "labels": [s["label"] for s in usable],
    }

"""
Credibility & Framing Lens.

A media-literacy tool, not a fact-checker: Gemini's free API tier does
not include live Google Search grounding (that requires a billing-
enabled project -- see https://ai.google.dev/gemini-api/docs/pricing),
so rather than faking "verification" this analyzes the *language* of a
piece of text for signals worth a reader's attention -- emotionally
loaded or absolutist wording, overall emotional tone, and reading
difficulty -- and extracts the concrete factual claims being made so the
reader can independently check the ones that matter, with a one-click
search link for each (a plain URL, not an API call -- free and instant).

This is squarely aimed at a real, current problem: most people have no
quick way to gauge whether something they're reading is written to
inform or to provoke. Loaded-language detection and readability scoring
are both plain, deterministic algorithms (no API call, no cost, no rate
limit); VADER (already used elsewhere in this app for YouTube comment
sentiment) scores overall emotional tone; only claim extraction calls
Gemini, and only once per analysis.
"""
from __future__ import annotations

import re
import urllib.parse

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient
from .readability import readability_scores

_analyzer = SentimentIntensityAnalyzer()

# Small, hand-curated lexicons -- not exhaustive, but enough to surface a
# meaningful signal. Grouped by the rhetorical pattern they flag.
_ABSOLUTIST = {
    "always", "never", "everyone", "no one", "nobody", "everybody",
    "completely", "totally", "absolutely", "undeniably", "unquestionably",
    "guaranteed", "impossible", "proof", "proves", "entirely",
}
_EMOTIONAL = {
    "shocking", "outrageous", "devastating", "horrifying", "terrifying",
    "disgraceful", "scandal", "catastrophe", "explosive", "bombshell",
    "slams", "blasts", "destroys", "obliterates", "crisis", "chaos",
    "furious", "shameful", "alarming", "meltdown",
}
_HEDGING = {
    "some say", "many believe", "it is said", "reportedly", "allegedly",
    "sources say", "critics say", "experts warn", "people are saying",
}
_CLICKBAIT_PATTERNS = (
    re.compile(r"\byou won.t believe\b", re.I),
    re.compile(r"\bwhat happened next\b", re.I),
    re.compile(r"\bnumber \d+ will\b", re.I),
    re.compile(r"\bthis one (trick|weird)\b", re.I),
    re.compile(r"\bdoctors hate\b", re.I),
)


class CredibilityError(RuntimeError):
    pass


def _find_terms(text_lower: str, terms: set[str]) -> list[str]:
    found = []
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
            found.append(term)
    return found


def loaded_language_report(text: str) -> dict:
    """Score how much emotionally loaded / absolutist / hedging language a
    text uses, per 100 words -- a rough proxy for persuasive or
    sensationalized framing versus neutral reporting."""
    text = text or ""
    words = text.split()
    word_count = max(len(words), 1)
    text_lower = text.lower()

    absolutist_hits = _find_terms(text_lower, _ABSOLUTIST)
    emotional_hits = _find_terms(text_lower, _EMOTIONAL)
    hedging_hits = _find_terms(text_lower, _HEDGING)
    clickbait_hits = [p.pattern for p in _CLICKBAIT_PATTERNS if p.search(text)]

    total_hits = len(absolutist_hits) + len(emotional_hits) + len(hedging_hits) + len(clickbait_hits)
    density = round((total_hits / word_count) * 100, 2)

    if density >= 3:
        level = "High"
    elif density >= 1:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "density_per_100_words": density,
        "level": level,
        "absolutist_terms": sorted(absolutist_hits),
        "emotional_terms": sorted(emotional_hits),
        "hedging_phrases": sorted(hedging_hits),
        "clickbait_patterns_matched": len(clickbait_hits),
    }


def sentiment_profile(text: str) -> dict:
    scores = _analyzer.polarity_scores(text or "")
    compound = scores["compound"]
    if compound >= 0.3:
        tone = "Strongly positive"
    elif compound >= 0.05:
        tone = "Leaning positive"
    elif compound <= -0.3:
        tone = "Strongly negative"
    elif compound <= -0.05:
        tone = "Leaning negative"
    else:
        tone = "Neutral"
    return {"tone": tone, "compound": round(compound, 3)}


def extract_claims(text: str, max_claims: int = 5) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    prompt = (
        f"List up to {max_claims} distinct, concrete, checkable factual "
        "claims made in the text below (things a reader could look up and "
        "confirm or refute) -- not opinions, predictions, or vague "
        "statements. One claim per line, no numbering, no preamble, no "
        "commentary. If there are no clear factual claims, return "
        "nothing.\n\n"
        f"TEXT:\n{text[:6000]}"
    )
    try:
        response = generate_content_resilient(prompt)
        raw = (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise CredibilityError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise CredibilityError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise CredibilityError(f"Claim extraction failed: {exc}") from exc

    claims = [line.strip("-* ").strip() for line in raw.splitlines() if line.strip()]
    return claims[:max_claims]


def _search_link(claim: str) -> str:
    return "https://www.google.com/search?q=" + urllib.parse.quote(claim)


def analyze_credibility(text: str) -> dict:
    """Combine loaded-language density, emotional tone, readability, and
    Gemini-extracted claims (each with a one-click verification search
    link) into one panel."""
    text = text or ""
    claims: list[str] = []
    claims_error = None
    try:
        claims = extract_claims(text)
    except CredibilityError as exc:
        claims_error = str(exc)

    return {
        "loaded_language": loaded_language_report(text),
        "sentiment": sentiment_profile(text),
        "readability": readability_scores(text),
        "claims": [{"text": c, "search_url": _search_link(c)} for c in claims],
        "claims_error": claims_error,
    }

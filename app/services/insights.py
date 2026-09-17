"""
The bridge between the algorithms in app/core/ and the user's own data.

Everything reachable from here runs locally on this machine. No request
leaves the process, no API key is required, and nothing here degrades
when Gemini is down, rate-limited, or discontinued. That is the point of
app/core/ existing: the parts of this product that are ours don't rent
their intelligence from anyone.

Three features live here:

*Provenance* -- for any summary, work out which passage of the original
each sentence came from, so a reader can check the claim instead of
trusting it. This is the honest answer to "how do I know the AI didn't
make this up".

*Timeline* -- pull every date expression out of everything you've read
and lay it on a chronology, resolving relative dates ("three weeks ago")
against the document that said them rather than against today.

*Contradictions* -- extract every quantity, normalise the units, group
claims about the same thing, and surface where your sources disagree.
Nobody notices that one report said 4.2 million and another said 6.1
million when they were read a month apart. A machine notices instantly.
"""
from __future__ import annotations

import logging

from ..core import provenance as provenance_core
from ..core import quantities as quantities_core
from ..core import temporal as temporal_core
from ..models import HistoryEntry

logger = logging.getLogger(__name__)

# Scanning a corpus is linear in total characters, and a user with years
# of history could have tens of megabytes of source text. These caps keep
# a page load predictable; the UI says plainly when a cap was hit rather
# than quietly analysing a subset and presenting it as the whole.
_MAX_DOCUMENTS = 400
_MAX_CHARS_PER_DOCUMENT = 200_000


def _entry_documents(user_id: int, *, limit: int = _MAX_DOCUMENTS, include_archived: bool = False) -> list[dict]:
    query = HistoryEntry.query.filter_by(user_id=user_id)
    if not include_archived:
        query = query.filter(HistoryEntry.is_archived.isnot(True))
    entries = query.order_by(HistoryEntry.id.desc()).limit(limit).all()

    documents = []
    for entry in entries:
        text = (entry.source_text or "") or (entry.summary or "")
        if not text.strip():
            continue
        documents.append(
            {
                "id": entry.id,
                "title": entry.display_title,
                "text": text[:_MAX_CHARS_PER_DOCUMENT],
                "created_at": entry.created_at.isoformat() if entry.created_at else "",
                "source_type": entry.source_type,
            }
        )
    return documents


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def trace_entry(entry_id: int, user_id: int) -> dict:
    """Align an entry's summary against its own source text.

    Returns a payload the entry page can render directly: one row per
    summary sentence with the character span it maps to, plus a coverage
    figure for the summary as a whole.

    A word on honesty: a "strong" match means the sentence is lexically
    present in the source, not that the sentence is *true*. A summary
    can recombine real phrases into a claim the source never made. The
    UI wording reflects that -- it says "found here", never "verified".
    """
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    if entry is None:
        return {"available": False, "reason": "not_found"}

    source_text = (entry.source_text or "").strip()
    summary = (entry.summary or "").strip()

    if not summary:
        return {"available": False, "reason": "no_summary"}
    if not source_text:
        # Audio, image and video entries summarize the media directly,
        # so there is no text original to point at. Saying so is better
        # than showing an empty table.
        return {"available": False, "reason": "no_source_text", "source_type": entry.source_type}

    try:
        alignments = provenance_core.align_summary(summary, source_text)
        coverage = provenance_core.coverage_report(summary, source_text)
    except Exception as exc:  # pragma: no cover
        logger.warning("Provenance alignment failed for entry %s: %s", entry_id, exc)
        return {"available": False, "reason": "error"}

    return {
        "available": True,
        "entry_id": entry.id,
        "title": entry.display_title,
        "source_text": source_text,
        "alignments": alignments,
        "coverage": coverage,
    }


def trace_text(summary: str, source_text: str) -> dict:
    """Ad-hoc alignment of any two pieces of text -- used by the "check
    this against that" tool, which works on text the user pastes in and
    never had to be summarized by us at all."""
    summary = (summary or "").strip()
    source_text = (source_text or "").strip()
    if not summary or not source_text:
        return {"available": False, "reason": "missing_input"}
    return {
        "available": True,
        "alignments": provenance_core.align_summary(summary, source_text),
        "coverage": provenance_core.coverage_report(summary, source_text),
        "source_text": source_text,
    }


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def build_timeline(user_id: int, *, limit: int = _MAX_DOCUMENTS, min_confidence: float = 0.4) -> dict:
    """Every date mentioned across the corpus, placed on one chronology.

    `min_confidence` filters out the deliberately vague matches the
    parser emits at low confidence ("recently", "a while back"). They're
    real expressions and worth extracting, but putting them on a timeline
    at a made-up position would be fiction.
    """
    documents = _entry_documents(user_id, limit=limit)
    if not documents:
        return {"events": [], "buckets": {"by_year": {}, "by_decade": {}}, "span": {"earliest": None, "latest": None}, "total": 0, "documents_scanned": 0}

    try:
        timeline = temporal_core.build_timeline(documents)
    except Exception as exc:  # pragma: no cover
        logger.warning("Timeline build failed: %s", exc)
        return {"events": [], "buckets": {"by_year": {}, "by_decade": {}}, "span": {"earliest": None, "latest": None}, "total": 0, "documents_scanned": len(documents), "error": True}

    events = [e for e in timeline.get("events", []) if e.get("confidence", 0) >= min_confidence]
    timeline["events"] = events
    timeline["total"] = len(events)
    timeline["documents_scanned"] = len(documents)
    timeline["truncated"] = len(documents) >= limit
    return timeline


def timeline_for_entry(entry_id: int, user_id: int) -> dict:
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    if entry is None:
        return {"events": [], "total": 0}
    text = (entry.source_text or "") or (entry.summary or "")
    documents = [
        {
            "id": entry.id,
            "title": entry.display_title,
            "text": text[:_MAX_CHARS_PER_DOCUMENT],
            "created_at": entry.created_at.isoformat() if entry.created_at else "",
        }
    ]
    return temporal_core.build_timeline(documents)


# ---------------------------------------------------------------------------
# Numeric claims and contradictions
# ---------------------------------------------------------------------------


def find_contradictions(user_id: int, *, limit: int = _MAX_DOCUMENTS) -> dict:
    """Where your sources disagree on a number.

    Currencies are never converted and never compared across currencies
    -- exchange rates are market prices that move, not definitions, so
    "$12M vs €12M" is not a contradiction and claiming otherwise would
    be wrong. See app/core/quantities.py for the full reasoning.
    """
    documents = _entry_documents(user_id, limit=limit)
    if not documents:
        return {"groups": [], "conflicts": [], "total_quantities": 0, "documents_scanned": 0}

    try:
        result = quantities_core.find_contradictions(documents)
    except Exception as exc:  # pragma: no cover
        logger.warning("Contradiction scan failed: %s", exc)
        return {"groups": [], "conflicts": [], "total_quantities": 0, "documents_scanned": len(documents), "error": True}

    result["documents_scanned"] = len(documents)
    result["truncated"] = len(documents) >= limit
    return result


def quantities_for_entry(entry_id: int, user_id: int) -> list[dict]:
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    if entry is None:
        return []
    text = (entry.source_text or "") or (entry.summary or "")
    try:
        return quantities_core.extract_quantities(text[:_MAX_CHARS_PER_DOCUMENT])
    except Exception:  # pragma: no cover
        return []


def corpus_overview(user_id: int) -> dict:
    """A cheap header for the insights pages: how much material these
    local analyses are actually working with."""
    total = HistoryEntry.query.filter_by(user_id=user_id).count()
    archived = HistoryEntry.query.filter_by(user_id=user_id, is_archived=True).count()
    return {"entries": total, "archived": archived, "active": total - archived}

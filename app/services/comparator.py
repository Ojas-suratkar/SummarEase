"""
Structured multi-document comparator.

Reading two or three sources and building your own mental (or literal)
comparison table -- what does each one actually say about price, what
does each claim about performance, where do they actually disagree -- is
exactly the kind of task a chat interface answers in prose, which you
then have to re-read three times and transcribe into a table yourself if
you actually want to compare cleanly. This does the transcription: Gemini
is constrained with a JSON schema (`response_json_schema`, the current
google-genai SDK's structured-output mechanism -- see
https://ai.google.dev/gemini-api/docs/structured-output) to return the
comparison as data, not prose, so the output is a real, consistently-
shaped table every time, not a chat answer that happens to have some
bullet points in it.

The model chooses which dimensions matter (price, release date, and
approach for products; sample size and methodology for studies; scope
and enforcement for two policies -- whatever's actually relevant to what
was given it) rather than this code assuming a fixed schema like "price /
rating / pros / cons" that wouldn't fit most comparisons.
"""
from __future__ import annotations

import json

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient

_MAX_SOURCES = 5
_MIN_SOURCES = 2

_COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "description": "The aspect being compared, e.g. 'Price' or 'Sample size'.",
                    },
                    "values": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "value": {
                                    "type": "string",
                                    "description": "A short answer for this source on this dimension. Use 'Not stated' if the source doesn't address it.",
                                },
                            },
                            "required": ["source", "value"],
                        },
                    },
                },
                "required": ["dimension", "values"],
            },
        },
        "agreements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Points where the sources substantively agree.",
        },
        "disagreements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Points where the sources substantively conflict or diverge.",
        },
    },
    "required": ["rows", "agreements", "disagreements"],
}


class ComparatorError(RuntimeError):
    pass


def compare_sources(sources: list[dict]) -> dict:
    """`sources` is a list of {"label": str, "text": str}. Returns a
    structured comparison: rows (one per dimension, one cell per source),
    plus explicit lists of where the sources agree and disagree."""
    sources = [s for s in (sources or []) if s.get("text", "").strip()]
    if len(sources) < _MIN_SOURCES:
        raise ComparatorError(f"Add at least {_MIN_SOURCES} sources with usable content to compare.")
    sources = sources[:_MAX_SOURCES]

    labels = [s["label"] for s in sources]
    blocks = "\n\n".join(
        f"=== SOURCE: {s['label']} ===\n{s['text'][:5000]}" for s in sources
    )
    prompt = (
        f"Compare these {len(sources)} sources ({', '.join(labels)}) side by "
        "side. First, decide which 4-8 dimensions are actually the most "
        "useful to compare them on, given what they're actually about "
        "(don't force a generic template -- e.g. compare products on price "
        "and features, studies on methodology and sample size, policies on "
        "scope and enforcement, arguments on their core claim and "
        "evidence). For every dimension, give a short value for every "
        "source, using exactly the source label given -- if a source "
        "doesn't address that dimension, say so rather than guessing. "
        "Then separately list concrete points where the sources agree, and "
        "concrete points where they disagree or conflict.\n\n"
        f"{blocks}"
    )
    from google.genai import types

    try:
        response = generate_content_resilient(
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_COMPARISON_SCHEMA,
            ),
        )
        data = json.loads(response.text)
    except GeminiNotConfiguredError as exc:
        raise ComparatorError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise ComparatorError(str(exc)) from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise ComparatorError(f"Couldn't parse the comparison from the model's response: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 -- network/API errors
        raise ComparatorError(f"Comparison failed: {exc}") from exc

    rows = []
    for row in data.get("rows", []):
        dimension = (row.get("dimension") or "").strip()
        if not dimension:
            continue
        values = {v.get("source", ""): v.get("value", "") for v in row.get("values", [])}
        rows.append({"dimension": dimension, "values": [values.get(label, "Not stated") for label in labels]})

    return {
        "labels": labels,
        "rows": rows,
        "agreements": [a for a in data.get("agreements", []) if a],
        "disagreements": [d for d in data.get("disagreements", []) if d],
    }

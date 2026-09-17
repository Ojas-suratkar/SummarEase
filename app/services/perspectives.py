"""
Steelmanned multi-perspective generator.

Ask a chat interface "what do people think about this" and you typically
get a wishy-washy "there are many viewpoints" hedge, or a strawmanned
version of the side the model likes least. This does something more
specific and more useful: it identifies the actual contested question a
source touches (if there is one), and then, for each major position,
generates the *strongest good-faith version* of that position -- its
best argument, the key assumption it rests on, and what evidence would
actually change a reasonable holder's mind. That last part is the point:
it's a falsifiability check baked into every perspective, so this can't
degrade into empty both-sidesism. If a source isn't actually about
anything contested, this says so rather than manufacturing a debate.
"""
from __future__ import annotations

import json

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient

_PERSPECTIVES_SCHEMA = {
    "type": "object",
    "properties": {
        "is_contested": {
            "type": "boolean",
            "description": "True only if the source actually engages a genuinely contested question.",
        },
        "topic": {
            "type": "string",
            "description": "The specific contested question, phrased neutrally. Empty if is_contested is false.",
        },
        "perspectives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Short neutral name for this position."},
                    "steelman": {
                        "type": "string",
                        "description": "The strongest good-faith case for this position, 2-4 sentences.",
                    },
                    "key_assumption": {
                        "type": "string",
                        "description": "The core assumption or value judgment this position depends on.",
                    },
                    "would_change_mind": {
                        "type": "string",
                        "description": "Concrete evidence or events that would reasonably change a thoughtful holder's mind.",
                    },
                },
                "required": ["label", "steelman", "key_assumption", "would_change_mind"],
            },
        },
    },
    "required": ["is_contested", "topic", "perspectives"],
}


class PerspectivesError(RuntimeError):
    pass


def generate_perspectives(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise PerspectivesError("No source text to analyze.")

    prompt = (
        "Read the text below. First decide whether it actually engages a "
        "genuinely contested question -- one where thoughtful, informed "
        "people reasonably disagree -- as opposed to being purely "
        "factual/technical/uncontroversial. If it's not contested, say so "
        "and return an empty perspectives list; do not invent a "
        "controversy that isn't there.\n\n"
        "If it IS contested, identify 2-4 major distinct positions on that "
        "question (not just 'for' and 'against' if the real landscape has "
        "more nuance than that). For each, write the STRONGEST good-faith "
        "version of that position -- the version its most thoughtful "
        "defenders would actually give, not a weak or exaggerated version "
        "of it. Also name the key assumption or value judgment it rests "
        "on, and concretely what evidence or outcome would reasonably "
        "change a thoughtful holder's mind (this must be specific and "
        "falsifiable, not 'if they saw more evidence').\n\n"
        f"TEXT:\n{text[:6000]}"
    )
    from google.genai import types

    try:
        response = generate_content_resilient(
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_PERSPECTIVES_SCHEMA,
            ),
        )
        data = json.loads(response.text)
    except GeminiNotConfiguredError as exc:
        raise PerspectivesError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise PerspectivesError(str(exc)) from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise PerspectivesError(f"Couldn't parse perspectives from the model's response: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 -- network/API errors
        raise PerspectivesError(f"Perspective generation failed: {exc}") from exc

    return {
        "is_contested": bool(data.get("is_contested")),
        "topic": data.get("topic", ""),
        "perspectives": [
            {
                "label": p.get("label", ""),
                "steelman": p.get("steelman", ""),
                "key_assumption": p.get("key_assumption", ""),
                "would_change_mind": p.get("would_change_mind", ""),
            }
            for p in data.get("perspectives", [])
            if p.get("label") and p.get("steelman")
        ],
    }

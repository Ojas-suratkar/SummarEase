"""
Spaced-repetition retention system.

A summary you read once and never see again is forgotten within days --
that's not a SummarEase problem, it's how memory works (the forgetting
curve). This turns any summarized source into a small deck of flashcards
and schedules their review with the SM-2 algorithm -- the same spaced-
repetition scheduler behind Anki and SuperMemo -- so what you actually
retain from something you read here compounds over time instead of
evaporating the moment you close the tab. No chat interface does this:
answering a question about a document is not the same as making sure you
still remember the answer three weeks from now.

SM-2, briefly: each card has an ease factor (how easy this card is for
you, starts at 2.5) and an interval (days until it's due again). Grading
a card 0-5 (again/hard/good/easy, mapped below) updates both -- a good
answer pushes the interval out further and further; a poor one resets it
to daily review. The formulas below are the standard SM-2 formulas
(Piotr Wozniak, 1987), not an approximation.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient
from ..extensions import db
from ..models import Flashcard

_MIN_EASE_FACTOR = 1.3
_DEFAULT_EASE_FACTOR = 2.5

_FLASHCARD_SCHEMA = {
    "type": "object",
    "properties": {
        "flashcards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        }
    },
    "required": ["flashcards"],
}


class SpacedRepetitionError(RuntimeError):
    pass


def _card_dict(c: Flashcard) -> dict:
    return {
        "id": c.id,
        "source_ref": c.source_ref,
        "question": c.question,
        "answer": c.answer,
        "ease_factor": c.ease_factor,
        "interval_days": c.interval_days,
        "repetitions": c.repetitions,
        "next_review_at": c.next_review_at.isoformat(),
        "last_reviewed_at": c.last_reviewed_at.isoformat() if c.last_reviewed_at else None,
        "created_at": c.created_at.isoformat(),
    }


def generate_flashcards(text: str, source_ref: str = "", count: int = 6) -> list[dict]:
    """Ask Gemini for `count` question/answer pairs that test genuine
    recall of the source's substance (not trivia about phrasing), using
    structured JSON output so parsing never depends on the model
    following an ad-hoc text format."""
    text = (text or "").strip()
    if not text:
        raise SpacedRepetitionError("No source text to generate flashcards from.")

    prompt = (
        f"Create exactly {count} flashcards (question + answer pairs) that "
        "test genuine understanding and recall of the key facts, figures, "
        "and ideas in the text below -- not trivia about specific wording. "
        "Questions should be answerable from the text alone, concise, and "
        "varied (not all the same pattern). Answers should be short and "
        "precise -- a sentence or a fact, not a paragraph.\n\n"
        f"TEXT:\n{text[:8000]}"
    )
    from google.genai import types

    try:
        response = generate_content_resilient(
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_FLASHCARD_SCHEMA,
            ),
        )
        data = json.loads(response.text)
    except GeminiNotConfiguredError as exc:
        raise SpacedRepetitionError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise SpacedRepetitionError(str(exc)) from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise SpacedRepetitionError(f"Couldn't parse flashcards from the model's response: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 -- network/API errors
        raise SpacedRepetitionError(f"Flashcard generation failed: {exc}") from exc

    cards = [
        {"question": c.get("question", "").strip(), "answer": c.get("answer", "").strip()}
        for c in data.get("flashcards", [])
        if c.get("question") and c.get("answer")
    ]
    if not cards:
        raise SpacedRepetitionError("The model didn't return any usable flashcards.")
    return cards[:count]


def save_flashcards(user_id: int, source_ref: str, cards: list[dict]) -> int:
    for c in cards:
        db.session.add(
            Flashcard(
                user_id=user_id,
                source_ref=source_ref,
                question=c["question"],
                answer=c["answer"],
                ease_factor=_DEFAULT_EASE_FACTOR,
            )
        )
    db.session.commit()
    return len(cards)


def generate_and_save(user_id: int, text: str, source_ref: str = "", count: int = 6) -> int:
    cards = generate_flashcards(text, source_ref, count)
    return save_flashcards(user_id, source_ref, cards)


def due_flashcards(user_id: int, limit: int = 20) -> list[dict]:
    now = datetime.now(timezone.utc)
    cards = (
        Flashcard.query.filter(Flashcard.user_id == user_id, Flashcard.next_review_at <= now)
        .order_by(Flashcard.next_review_at.asc())
        .limit(limit)
        .all()
    )
    return [_card_dict(c) for c in cards]


def due_count(user_id: int) -> int:
    now = datetime.now(timezone.utc)
    return Flashcard.query.filter(Flashcard.user_id == user_id, Flashcard.next_review_at <= now).count()


def total_count(user_id: int) -> int:
    return Flashcard.query.filter_by(user_id=user_id).count()


def _sm2(ease_factor: float, interval_days: float, repetitions: int, quality: int) -> tuple[float, float, int]:
    """The standard SM-2 update. `quality` is 0-5 (0 = total blackout,
    5 = perfect recall). Returns (new_ease_factor, new_interval_days,
    new_repetitions)."""
    if quality < 3:
        # Failed recall -- reset the learning streak and review again
        # tomorrow, but don't touch the ease factor down here; SM-2
        # still updates ease factor even on a fail (below), it just also
        # resets the interval/repetition streak.
        repetitions = 0
        interval_days = 1
    else:
        if repetitions == 0:
            interval_days = 1
        elif repetitions == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease_factor, 2)
        repetitions += 1

    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease_factor = max(_MIN_EASE_FACTOR, ease_factor)

    return round(ease_factor, 3), interval_days, repetitions


def grade_flashcard(card_id: int, user_id: int, quality: int) -> dict:
    """Grade a card 0-5 and reschedule it per SM-2. Returns the card's
    updated scheduling state."""
    quality = max(0, min(5, int(quality)))
    card = Flashcard.query.filter_by(id=card_id, user_id=user_id).first()
    if card is None:
        raise SpacedRepetitionError("Flashcard not found.")

    new_ease, new_interval, new_reps = _sm2(
        card.ease_factor, card.interval_days or 1, card.repetitions, quality
    )
    next_review = datetime.now(timezone.utc) + timedelta(days=new_interval)

    card.ease_factor = new_ease
    card.interval_days = new_interval
    card.repetitions = new_reps
    card.next_review_at = next_review
    card.last_reviewed_at = datetime.now(timezone.utc)
    db.session.commit()

    return {
        "id": card_id,
        "ease_factor": new_ease,
        "interval_days": new_interval,
        "repetitions": new_reps,
        "next_review_at": next_review.isoformat(),
    }

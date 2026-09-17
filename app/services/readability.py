"""
Classic readability formulas (Flesch Reading Ease, Flesch-Kincaid Grade
Level) -- deterministic, no API call, no dependency beyond the standard
library and `re`. Used to flag text that's unusually dense, and as part
of the Credibility & Framing Lens (credibility_lens.py).
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def _count_syllables(word: str) -> int:
    word = word.lower()
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_scores(text: str) -> dict:
    text = (text or "").strip()
    words = _WORD_RE.findall(text)
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    if word_count == 0:
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "level": "N/A",
            "word_count": 0,
            "sentence_count": 0,
        }

    syllable_count = sum(_count_syllables(w) for w in words)
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count

    reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    grade_level = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59

    if reading_ease >= 70:
        level = "Easy"
    elif reading_ease >= 50:
        level = "Standard"
    elif reading_ease >= 30:
        level = "Difficult"
    else:
        level = "Very difficult"

    return {
        "flesch_reading_ease": round(reading_ease, 1),
        "flesch_kincaid_grade": round(max(grade_level, 0), 1),
        "level": level,
        "word_count": word_count,
        "sentence_count": sentence_count,
    }

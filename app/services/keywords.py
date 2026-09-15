"""
Keyword extraction using TF-IDF over the sentences of a single document.

Treating each sentence as a "document" lets scikit-learn's TF-IDF weighting
surface terms that are locally important without needing a large reference
corpus.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return sentences if len(sentences) >= 2 else [text]


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Return up to `top_n` keywords/phrases ranked by TF-IDF weight."""
    text = (text or "").strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 2),
            max_features=200,
        )
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Happens if every token is a stop word / too short a document.
        return []

    scores = matrix.max(axis=0).toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda pair: pair[1], reverse=True)

    keywords: list[str] = []
    seen_words: set[str] = set()
    for term, score in ranked:
        if score <= 0:
            continue
        words = set(term.split())
        if words & seen_words:
            # Skip near-duplicates like "data" after "data science" was kept.
            continue
        keywords.append(term)
        seen_words |= words
        if len(keywords) == top_n:
            break
    return keywords

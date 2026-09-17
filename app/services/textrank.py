"""
TextRank: a graph-centrality algorithm (the same family as PageRank) for
extractive summarization -- ranks sentences by how similar they are to
the rest of the document, without calling an LLM. Used as a pre-filter
in summarizer.summarize_adaptive: on long documents, only the most
salient fraction of sentences gets sent to Gemini for the abstractive
pass, which is faster, cheaper, and more consistent than truncating or
blindly chunking the raw text.

Implemented directly with TF-IDF (scikit-learn, already a dependency of
this app for keyword extraction) and a numpy power-iteration eigenvector
solve, rather than pulling in a graph library -- for one document's
worth of sentences this converges in a handful of iterations and needs
nothing heavier than numpy.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DAMPING = 0.85
_MAX_ITERATIONS = 50
_TOLERANCE = 1e-4


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split((text or "").strip()) if s.strip()]


def _pagerank(similarity_matrix: np.ndarray) -> np.ndarray:
    """Power-iteration PageRank over a sentence-similarity graph."""
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    matrix = similarity_matrix.copy()
    np.fill_diagonal(matrix, 0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = matrix / row_sums

    scores = np.full(n, 1.0 / n)
    for _ in range(_MAX_ITERATIONS):
        new_scores = (1 - _DAMPING) / n + _DAMPING * (transition.T @ scores)
        if np.abs(new_scores - scores).sum() < _TOLERANCE:
            scores = new_scores
            break
        scores = new_scores
    return scores


def rank_sentences(sentences: list[str]) -> np.ndarray:
    """Return a TextRank centrality score for each sentence, same order
    as the input."""
    if len(sentences) < 2:
        return np.ones(len(sentences))
    try:
        vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # All-stopword / degenerate input -- treat every sentence equally.
        return np.ones(len(sentences))

    similarity_matrix = cosine_similarity(matrix)
    return _pagerank(similarity_matrix)


def extractive_summary(text: str, ratio: float = 0.4, min_sentences: int = 3) -> list[str]:
    """Return the most salient sentences from `text`, in their original
    order, keeping roughly `ratio` of them (never fewer than
    `min_sentences` if the document has at least that many)."""
    sentences = split_sentences(text)
    if len(sentences) <= min_sentences:
        return sentences

    scores = rank_sentences(sentences)
    keep = max(min_sentences, round(len(sentences) * ratio))
    keep = min(keep, len(sentences))
    top_indices = set(np.argsort(-scores)[:keep].tolist())
    return [s for i, s in enumerate(sentences) if i in top_indices]

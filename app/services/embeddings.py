"""
Text embeddings + cosine-similarity vector search.

This is the retrieval algorithm behind two features: the semantic
Knowledge Base (history_store.py) and per-document "Ask this document"
Q&A (rag.py). Rather than standing up a separate vector database,
embeddings are plain numpy arrays and retrieval is exact cosine-
similarity ranking done in-process -- fast and exact at the scale a
single local user's history or one document's chunks will ever reach
(thousands of vectors, not millions).

Vectors are unit-normalized on the way out, so cosine similarity reduces
to a plain dot product (see `cosine_similarities`).
"""
from __future__ import annotations

import numpy as np

from .cache import embedding_cache_get_or_set
from .gemini_client import EMBEDDING_MODEL_NAME, GeminiNotConfiguredError, GeminiUnavailableError, embed_content_resilient

# gemini-embedding-001 produces Matryoshka-style embeddings: a shorter
# prefix of the full vector is still a valid, useful embedding at that
# length. 768 dims is plenty for cosine-similarity search at this app's
# scale and keeps stored vectors small (a few KB each).
DIMENSIONS = 768


class EmbeddingError(RuntimeError):
    """Raised when a piece of text could not be embedded."""


def _raw_embed_values(text: str, task_type: str) -> list[float]:
    """Uncached embedding call, resilient across a fallback chain of
    embedding models (see gemini_client.embed_content_resilient) -- returns
    plain floats (JSON-serializable) for the cache layer to store."""
    from google.genai import types

    try:
        result = embed_content_resilient(
            text,
            config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS, task_type=task_type),
        )
    except TypeError:
        # SDK signature drift -- fall back to a bare call without the
        # optional config rather than failing outright.
        result = embed_content_resilient(text)
    return list(result.embeddings[0].values)


def _raw_embed(text: str, task_type: str) -> np.ndarray:
    # Embeddings are a pure function of (model, task_type, text) -- cache
    # them the same way summarize_text is cached, so re-indexing the same
    # document or re-asking the same question doesn't re-hit the API.
    values, _cache_hit = embedding_cache_get_or_set(
        "embed_text", [EMBEDDING_MODEL_NAME, task_type, text], lambda: _raw_embed_values(text, task_type)
    )
    return np.array(values, dtype=np.float32)


def _normalize_dims(vector: np.ndarray) -> np.ndarray:
    """Force any returned vector to exactly DIMENSIONS entries, truncating
    (safe for Matryoshka-style embeddings) or zero-padding as needed."""
    if vector.shape[0] > DIMENSIONS:
        vector = vector[:DIMENSIONS]
    elif vector.shape[0] < DIMENSIONS:
        vector = np.pad(vector, (0, DIMENSIONS - vector.shape[0]))
    return vector


def embed_text(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> np.ndarray:
    """Embed a single piece of text into a unit-normalized DIMENSIONS-length
    vector. `task_type` tunes the embedding for how it'll be used:
    SEMANTIC_SIMILARITY (general), RETRIEVAL_DOCUMENT (indexing a
    passage), or RETRIEVAL_QUERY (a search question)."""
    text = (text or "").strip()
    if not text:
        return np.zeros(DIMENSIONS, dtype=np.float32)

    try:
        vector = _normalize_dims(_raw_embed(text, task_type))
    except GeminiNotConfiguredError as exc:
        raise EmbeddingError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise EmbeddingError(f"Embedding failed: {exc}") from exc

    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


def embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Embed multiple texts; returns an (N, DIMENSIONS) matrix. This calls
    the API once per text -- fine at the scale this app operates at (a
    handful to a few dozen chunks per document)."""
    if not texts:
        return np.zeros((0, DIMENSIONS), dtype=np.float32)
    return np.vstack([embed_text(t, task_type=task_type) for t in texts])


def cosine_similarities(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query_vector: (DIMENSIONS,); matrix: (N, DIMENSIONS). Both are
    assumed already unit-normalized (embed_text/embed_batch do this), so
    cosine similarity is just the dot product."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query_vector


def top_k(query_vector: np.ndarray, matrix: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return up to k (index, similarity) pairs, highest similarity first."""
    sims = cosine_similarities(query_vector, matrix)
    if sims.size == 0:
        return []
    k = min(k, sims.size)
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]

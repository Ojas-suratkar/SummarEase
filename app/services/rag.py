"""
"Ask this document" -- retrieval-augmented Q&A grounded in one specific
piece of source content (whatever the user just summarized), not the
model's general knowledge. Also backs the Credibility Lens, Faithfulness
Check, flashcard generation, and Perspectives, all of which need the
document's full original text.

How it works: the source text is split into overlapping passages, each
passage is embedded once when the document is indexed, and each question
is answered by embedding the question, retrieving the most similar
passages via cosine similarity (embeddings.top_k), and asking Gemini to
answer using only those passages -- with an explicit instruction to say
so if the passages don't contain the answer, rather than filling gaps
from its own training data. This is a real (if small) retrieval-
augmented-generation pipeline, not just "paste the whole document into
the prompt every time."

A "document" here IS a knowledge-base entry -- `doc_id` is just
`str(HistoryEntry.id)`, not a separate identity. This used to be two
disconnected things: history_store.py durably saved a 500-character
excerpt, while this module cached the *full* text plus its chunk
embeddings in a plain in-memory dict, capped and time-limited (2 hours).
That meant Ask/Credibility/Faithfulness/Flashcards/Perspectives all quietly
stopped working on anything older than a couple of hours or across a
server restart -- the exact "sessions don't get stored" bug. Now the full
text is a durable column on HistoryEntry, and chunk embeddings
(RagChunk) are a persisted cache keyed by entry id, rebuilt on demand
from that column if they're ever missing rather than expiring.
"""
from __future__ import annotations

import numpy as np

from .embeddings import EmbeddingError, embed_batch, embed_text, top_k
from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient
from ..extensions import db
from ..models import HistoryEntry, RagChunk

_CHUNK_WORDS = 180
_CHUNK_OVERLAP = 40


class RagError(RuntimeError):
    pass


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(_CHUNK_WORDS - _CHUNK_OVERLAP, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + _CHUNK_WORDS])
        if chunk:
            chunks.append(chunk)
        if start + _CHUNK_WORDS >= len(words):
            break
    return chunks


def index_entry(entry_id: int, text: str) -> None:
    """Chunk + embed `text` and persist it as this entry's RAG index.
    Best-effort: called right after the entry is saved, and a failure
    here (e.g. no API key) just means Ask/Credibility/etc. won't work for
    this entry -- it doesn't undo the save. Safe to call again (e.g. from
    `_ensure_chunks` below) -- clears any existing chunks first."""
    text = (text or "").strip()
    if not text:
        return

    chunks = _chunk_text(text)
    if not chunks:
        return

    matrix = embed_batch(chunks, task_type="RETRIEVAL_DOCUMENT")  # raises EmbeddingError

    RagChunk.query.filter_by(entry_id=entry_id).delete()
    for i, (chunk_text, vector) in enumerate(zip(chunks, matrix)):
        db.session.add(
            RagChunk(
                entry_id=entry_id,
                chunk_index=i,
                text=chunk_text,
                embedding=np.asarray(vector, dtype=np.float32).tobytes(),
            )
        )
    db.session.commit()


def _ensure_chunks(entry_id: int, full_text: str) -> list[RagChunk]:
    """Return this entry's persisted chunks, rebuilding them from
    `full_text` if they're missing (e.g. the entry predates this table,
    or indexing failed at save time and a later retry succeeds)."""
    chunks = RagChunk.query.filter_by(entry_id=entry_id).order_by(RagChunk.chunk_index).all()
    if chunks:
        return chunks
    try:
        index_entry(entry_id, full_text)
    except EmbeddingError as exc:
        raise RagError(f"Could not index this document for Q&A: {exc}") from exc
    return RagChunk.query.filter_by(entry_id=entry_id).order_by(RagChunk.chunk_index).all()


def get_full_text(user_id: int, doc_id: str | int | None) -> str | None:
    if not doc_id:
        return None
    try:
        entry_id = int(doc_id)
    except (TypeError, ValueError):
        return None
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    return entry.source_text if entry else None


def ask(user_id: int, doc_id: str, question: str, top_n: int = 4) -> dict:
    question = (question or "").strip()
    if not question:
        raise RagError("Please ask a question.")

    full_text = get_full_text(user_id, doc_id)
    if full_text is None:
        raise RagError(
            "This document is no longer available to ask questions about "
            "-- try summarizing it again."
        )

    chunks = _ensure_chunks(int(doc_id), full_text)
    if not chunks:
        raise RagError("Couldn't index this document for Q&A.")

    matrix = np.vstack([np.frombuffer(c.embedding, dtype=np.float32) for c in chunks])

    try:
        query_vector = embed_text(question, task_type="RETRIEVAL_QUERY")
    except EmbeddingError as exc:
        raise RagError(f"Could not process your question: {exc}") from exc

    ranked = top_k(query_vector, matrix, k=top_n)
    if not ranked:
        raise RagError("Couldn't find anything relevant to that question in this document.")

    passages = [(chunks[idx].text, score) for idx, score in ranked]
    context = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{passage}" for i, (passage, _score) in enumerate(passages)
    )

    prompt = (
        "Answer the question using ONLY the passages below, which are "
        "excerpts from a single source document. If the passages don't "
        "contain enough information to answer, say so plainly instead of "
        "guessing or using outside knowledge.\n\n"
        f"{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer in 2-4 sentences, and mention which passage number(s) you "
        "drew on."
    )

    try:
        response = generate_content_resilient(prompt)
        answer = (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise RagError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise RagError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/API errors
        raise RagError(f"Couldn't generate an answer: {exc}") from exc

    return {
        "answer": answer,
        "passages": [
            {"text": passage, "similarity": round(score, 3)} for passage, score in passages
        ],
    }

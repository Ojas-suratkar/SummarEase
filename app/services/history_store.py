"""
Persistent knowledge base of everything a user has summarized, searchable
semantically (by meaning, not just keyword match) via text embeddings.

Design notes
------------
Backed by SQLAlchemy (`app/models.py`) instead of a hand-rolled sqlite3
connection -- SQLite locally, Postgres once deployed, same code either
way. Every function here takes a `user_id` and every query is scoped to
it, so one account's knowledge base is never visible to another.

`source_text` is now stored in full (see `HistoryEntry` in models.py for
why that matters -- it's the fix for documents becoming permanently
unavailable for Q&A/credibility/flashcards/perspectives after a restart).
Embeddings are stored as raw float32 bytes in a BLOB/BYTEA column and
loaded back into numpy arrays for cosine-similarity search (see
embeddings.py) -- no separate vector database needed at this scale.

Saving is always best-effort from the caller's side: routes.py wraps
every call here so that a knowledge-base failure (e.g. no API key
configured yet, so embeddings can't be produced) never breaks the
underlying summarize/PDF/article/YouTube request that triggered it.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .embeddings import EmbeddingError, embed_text, top_k
from ..extensions import db
from ..models import HistoryEntry, Link


class HistoryStoreError(RuntimeError):
    pass


def _entry_dict(entry: HistoryEntry, *, with_source_text: bool = False) -> dict:
    data = {
        "id": entry.id,
        "source_type": entry.source_type,
        "source_ref": entry.source_ref,
        "summary": entry.summary,
        "excerpt": entry.excerpt,
        "created_at": entry.created_at.isoformat(),
        "embedding": entry.embedding,
    }
    if with_source_text:
        data["source_text"] = entry.source_text or ""
    return data


def save_entry(user_id: int, source_type: str, source_ref: str, summary: str, source_text: str = "") -> int:
    """Persist one summary to the knowledge base. Embedding failures
    don't stop the entry being saved -- it just becomes unsearchable
    semantically, but still shows up in the recent list."""
    embedding_bytes = None
    try:
        vector = embed_text((summary or source_text)[:2000])
        embedding_bytes = vector.astype(np.float32).tobytes()
    except EmbeddingError:
        pass

    entry = HistoryEntry(
        user_id=user_id,
        source_type=source_type,
        source_ref=source_ref,
        summary=summary,
        source_text=source_text or "",
        embedding=embedding_bytes,
    )
    db.session.add(entry)
    db.session.commit()
    return entry.id


def list_recent(user_id: int, limit: int = 30) -> list[dict]:
    entries = (
        HistoryEntry.query.filter_by(user_id=user_id)
        .order_by(HistoryEntry.id.desc())
        .limit(limit)
        .all()
    )
    return [_entry_dict(e) for e in entries]


def search_history(user_id: int, query: str, top_n: int = 10) -> list[dict]:
    """Semantic search across everything a user has summarized, ranked by
    cosine similarity between the query and each stored entry's
    embedding."""
    query = (query or "").strip()
    if not query:
        return []

    try:
        query_vector = embed_text(query)
    except EmbeddingError as exc:
        raise HistoryStoreError(f"Semantic search unavailable: {exc}") from exc

    entries = HistoryEntry.query.filter(
        HistoryEntry.user_id == user_id, HistoryEntry.embedding.isnot(None)
    ).all()
    if not entries:
        return []

    matrix = np.vstack([np.frombuffer(e.embedding, dtype=np.float32) for e in entries])
    ranked = top_k(query_vector, matrix, k=top_n)

    results = []
    for idx, score in ranked:
        d = _entry_dict(entries[idx])
        d["similarity"] = round(score, 3)
        results.append(d)
    return results


def keyword_search(user_id: int, query: str, limit: int = 8) -> list[dict]:
    """Plain SQL LIKE search over summary/source_ref -- used by the
    command palette (see static/js/app.js) and the full-text search
    page, which need an instant, local, no-API-call result as you type.
    Deliberately separate from search_history's embedding-based semantic
    search, which is more powerful but costs a Gemini call."""
    query = (query or "").strip()
    if not query:
        return []
    like = f"%{query}%"
    entries = (
        HistoryEntry.query.filter(
            HistoryEntry.user_id == user_id,
            db.or_(HistoryEntry.summary.ilike(like), HistoryEntry.source_ref.ilike(like)),
        )
        .order_by(HistoryEntry.id.desc())
        .limit(limit)
        .all()
    )
    return [_entry_dict(e) for e in entries]


def entry_count(user_id: int) -> int:
    return HistoryEntry.query.filter_by(user_id=user_id).count()


def raw_clusters(user_id: int, n_clusters: int = 5) -> list[dict]:
    """Group every embedded knowledge-base entry into `n_clusters` topic
    clusters with k-means over their embeddings (unsupervised learning
    over the user's own reading history -- no LLM call here). Labels are
    added separately by topics.py, the only part of this feature that
    calls Gemini."""
    entries = HistoryEntry.query.filter(
        HistoryEntry.user_id == user_id, HistoryEntry.embedding.isnot(None)
    ).all()
    if not entries:
        return []

    matrix = np.vstack([np.frombuffer(e.embedding, dtype=np.float32) for e in entries])
    k = max(1, min(n_clusters, len(entries)))

    if k == 1:
        labels = np.zeros(len(entries), dtype=int)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(matrix)

    clusters: dict[int, list[dict]] = {}
    for entry, label in zip(entries, labels):
        clusters.setdefault(int(label), []).append(_entry_dict(entry))

    return [
        {"items": items} for _cid, items in sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    ]


def get_entry(entry_id: int) -> dict | None:
    """Unscoped lookup by id -- deliberately does *not* take a user_id.
    Used only where ownership is checked some other way: public share
    resolution (sharing.py, where the whole point is a link that works
    without being logged in as the owner) and the knowledge-graph
    auto-linker (which compares against a user's own other entries, not
    across accounts -- see other_embedded_entries)."""
    entry = HistoryEntry.query.get(entry_id)
    return _entry_dict(entry, with_source_text=True) if entry else None


def get_entry_for_user(entry_id: int, user_id: int) -> dict | None:
    """Ownership-checked lookup -- use this one from routes."""
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    return _entry_dict(entry, with_source_text=True) if entry else None


def all_embedded_entries(user_id: int) -> list[dict]:
    """Every embedded entry for this user -- used by dedup.py, which does
    its own pairwise similarity comparison over entries already embedded
    (no new Gemini calls needed for detection itself)."""
    entries = (
        HistoryEntry.query.filter(HistoryEntry.user_id == user_id, HistoryEntry.embedding.isnot(None))
        .order_by(HistoryEntry.id.asc())
        .all()
    )
    return [_entry_dict(e) for e in entries]


def delete_entry(entry_id: int, user_id: int) -> None:
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    if entry is None:
        return
    Link.query.filter(
        db.or_(Link.from_id == entry_id, Link.to_id == entry_id), Link.user_id == user_id
    ).delete()
    db.session.delete(entry)
    db.session.commit()


def repoint_links(old_id: int, new_id: int, user_id: int) -> None:
    """Used when merging duplicate entries (dedup.py) -- before deleting
    the duplicate, re-point any links that referenced it to the surviving
    entry instead, so the knowledge graph doesn't lose those relationships."""
    Link.query.filter_by(from_id=old_id, user_id=user_id).update({"from_id": new_id})
    Link.query.filter_by(to_id=old_id, user_id=user_id).update({"to_id": new_id})
    Link.query.filter(Link.from_id == Link.to_id, Link.user_id == user_id).delete(
        synchronize_session=False
    )
    db.session.commit()


def other_embedded_entries(user_id: int, exclude_id: int) -> list[dict]:
    """Every embedded entry for this user except `exclude_id` -- the
    candidate pool for auto-linking a newly saved entry."""
    entries = HistoryEntry.query.filter(
        HistoryEntry.user_id == user_id,
        HistoryEntry.embedding.isnot(None),
        HistoryEntry.id != exclude_id,
    ).all()
    return [_entry_dict(e) for e in entries]


def add_link(user_id: int, from_id: int, to_id: int, relationship: str, rationale: str = "", similarity: float = 0.0) -> int:
    link = Link(
        user_id=user_id,
        from_id=from_id,
        to_id=to_id,
        relationship_type=relationship,
        rationale=rationale,
        similarity=similarity,
    )
    db.session.add(link)
    db.session.commit()
    return link.id


def get_links_for(user_id: int, entry_id: int) -> list[dict]:
    """Every link touching `entry_id`, from either direction, joined with
    the *other* entry's summary/source info so the caller doesn't need a
    second query."""
    links = (
        Link.query.filter(
            Link.user_id == user_id, db.or_(Link.from_id == entry_id, Link.to_id == entry_id)
        )
        .order_by(Link.id.desc())
        .all()
    )

    results = []
    for link in links:
        other_id = link.to_id if link.from_id == entry_id else link.from_id
        other = HistoryEntry.query.filter_by(id=other_id, user_id=user_id).first()
        if other is None:
            continue
        results.append(
            {
                "link_id": link.id,
                "relationship": link.relationship_type,
                "rationale": link.rationale,
                "similarity": link.similarity,
                "outgoing": link.from_id == entry_id,
                **_entry_dict(other),
            }
        )
    return results


def graph_data(user_id: int, limit: int = 200) -> dict:
    """All entries + all links for this user, for the whole-graph view
    (/graph). Capped at `limit` most-recent entries so a large knowledge
    base still renders a readable picture."""
    entries = (
        HistoryEntry.query.filter_by(user_id=user_id)
        .order_by(HistoryEntry.id.desc())
        .limit(limit)
        .all()
    )
    entry_ids = {e.id for e in entries}
    if not entry_ids:
        return {"nodes": [], "edges": []}

    links = Link.query.filter(
        Link.user_id == user_id, Link.from_id.in_(entry_ids), Link.to_id.in_(entry_ids)
    ).all()

    nodes = [
        {"id": e.id, "source_type": e.source_type, "source_ref": e.source_ref, "summary": e.summary, "created_at": e.created_at.isoformat()}
        for e in entries
    ]
    edges = [{"from_id": link.from_id, "to_id": link.to_id, "relationship": link.relationship_type} for link in links]
    return {"nodes": nodes, "edges": edges}

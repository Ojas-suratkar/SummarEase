"""
Content-hash based response cache for Gemini calls that are pure
functions of their input -- summarize_text, embed_text, the weekly
digest narrative. Backed by the same database as everything else now
(models.py's `CacheEntry`) so it survives server restarts: re-summarizing
the same paste, or re-asking an identical question, returns instantly on
a cache hit and costs no API quota.

Deliberately global, not per-user (see CacheEntry's docstring in
models.py) -- the cache key already hashes the exact input, and two
different accounts summarizing the identical text get the identical
result from Gemini regardless of who asked, so sharing cache hits across
accounts saves calls without leaking anything (only the computed result
is stored, never who requested it or for which account).

Not used for anything where the call has side effects or depends on more
than its literal arguments (claim extraction against arbitrary-length
text, synthesis, RAG answers grounded in retrieved passages) -- caching
those would be either useless (near-zero hit rate) or subtly wrong.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Callable

from ..extensions import db
from ..models import CacheEntry

# In-memory, process-lifetime hit/miss counters -- surfaced on the ops
# dashboard (routes.py's /dashboard). Deliberately not persisted: this is
# a "how's it doing right now" signal, not historical data.
_stats_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0}


def get_stats() -> dict:
    with _stats_lock:
        hits, misses = _stats["hits"], _stats["misses"]
    total = hits + misses
    return {"hits": hits, "misses": misses, "hit_rate": round(hits / total, 3) if total else None}


# A week is long enough to make a multi-day demo/dev cycle fast, short
# enough that nothing sticks around forever if the underlying model or
# prompt changes.
DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7


def _make_key(namespace: str, parts: list[str]) -> str:
    raw = namespace + "\x1f" + "\x1f".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_call(namespace: str, parts: list[str], compute: Callable[[], str], ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Return (value, was_cache_hit). `compute` must return a plain string
    (every current use case -- a summary or a serialized embedding --
    fits that; keep it that way rather than generalizing prematurely)."""
    key = _make_key(namespace, parts)

    row = CacheEntry.query.get(key)
    if row is not None and time.time() - row.created_at < ttl_seconds:
        with _stats_lock:
            _stats["hits"] += 1
        return row.value, True

    result = compute()

    if row is not None:
        row.value = result
        row.created_at = time.time()
    else:
        db.session.add(CacheEntry(cache_key=key, value=result, created_at=time.time()))
    db.session.commit()

    with _stats_lock:
        _stats["misses"] += 1
    return result, False


def embedding_cache_get_or_set(namespace: str, parts: list[str], compute: Callable[[], list[float]]) -> tuple[list[float], bool]:
    """Same idea as `cached_call`, specialized for embedding vectors
    (stored as JSON arrays of floats rather than plain text)."""
    value_str, was_hit = cached_call(namespace, parts, lambda: json.dumps(compute()))
    return json.loads(value_str), was_hit

"""
Real local search engine over your own knowledge base -- the "not just
an LLM wrapper" feature the search page is built on.

The keyword half needs no API call at all: a plain SQL LIKE scan over
everything you've summarized (history_store.keyword_search), which is
instant and works even with no Gemini key configured or Gemini fully
down. The semantic half layers on top when embeddings are available
(history_store.search_history) so a search for "the president's economic
plan" can still surface something that never uses those literal words --
but it is an enhancement, not the foundation: this page works, and
returns real results, with zero calls to any third-party AI API.

Results from both are merged by entry id (a hit found by both counts as
stronger than a hit found by either alone) and returned as one ranked
list.
"""
from __future__ import annotations

from . import history_store


def search(user_id: int, query: str, limit: int = 20) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    by_id: dict[int, dict] = {}

    for r in history_store.keyword_search(user_id, query, limit=limit):
        by_id[r["id"]] = {**r, "match_type": "keyword", "score": 1.0}

    try:
        for r in history_store.search_history(user_id, query, top_n=limit):
            similarity = r.get("similarity", 0.0)
            if r["id"] in by_id:
                by_id[r["id"]]["match_type"] = "keyword + semantic"
                by_id[r["id"]]["score"] = 1.0 + similarity
                by_id[r["id"]]["similarity"] = similarity
            else:
                by_id[r["id"]] = {**r, "match_type": "semantic", "score": similarity}
    except history_store.HistoryStoreError:
        # No API key / embeddings unavailable -- keyword results above are
        # still a perfectly real result set, so this degrades quietly
        # rather than failing the whole search.
        pass

    results = sorted(by_id.values(), key=lambda r: -r["score"])
    return results[:limit]

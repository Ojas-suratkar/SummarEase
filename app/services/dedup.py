"""
Duplicate detection -- data hygiene, not analysis.

Summarize the same article from two different URLs (a syndicated wire
story, an AMP version and the original, a PDF you downloaded twice) and
your knowledge base quietly accumulates near-identical entries that
pollute search results and topic clusters without you noticing. This
scans every embedded entry pairwise (reusing embeddings already computed
and stored -- no new Gemini calls) for cosine similarity above a strict
threshold, groups the matches, and lets you merge each group down to one
entry with a click.

The threshold (0.92) is deliberately high -- well above what
knowledge_graph.py uses to *link* related entries (0.55). Linking wants
"these are meaningfully connected"; dedup wants "these are almost
certainly the same thing," so false positives here (which would delete
real data) need to be much rarer than false positives there (which just
mean a link that's a bit of a stretch).
"""
from __future__ import annotations

import numpy as np

from . import history_store
from .embeddings import cosine_similarities

DUPLICATE_THRESHOLD = 0.92


def find_duplicate_groups(user_id: int, threshold: float = DUPLICATE_THRESHOLD) -> list[list[dict]]:
    """Returns groups of 2+ entries whose embeddings are near-identical,
    highest-similarity groups first. Union-find over pairwise similarity
    above `threshold` -- deliberately simple (this app's knowledge base is
    at most a few thousand entries, so an O(n^2) pairwise pass is fine)."""
    entries = history_store.all_embedded_entries(user_id)
    n = len(entries)
    if n < 2:
        return []

    matrix = np.vstack([np.frombuffer(e["embedding"], dtype=np.float32) for e in entries])

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    pair_similarities: dict[tuple[int, int], float] = {}
    for i in range(n):
        sims = cosine_similarities(matrix[i], matrix[i + 1 :])
        for offset, sim in enumerate(sims):
            j = i + 1 + offset
            if sim >= threshold:
                union(i, j)
                pair_similarities[(i, j)] = float(sim)

    groups_by_root: dict[int, list[int]] = {}
    for i in range(n):
        groups_by_root.setdefault(find(i), []).append(i)

    groups = []
    for indices in groups_by_root.values():
        if len(indices) < 2:
            continue
        group_entries = [entries[i] for i in indices]
        # Best similarity involving any member of this group, for display.
        best_sim = max(
            (sim for (a, b), sim in pair_similarities.items() if a in indices or b in indices),
            default=threshold,
        )
        groups.append({"entries": group_entries, "similarity": round(best_sim, 3)})

    groups.sort(key=lambda g: -g["similarity"])
    return groups


def merge_group(user_id: int, keep_id: int, duplicate_ids: list[int]) -> int:
    """Delete `duplicate_ids`, keeping `keep_id` -- any knowledge-graph
    links that pointed at a deleted duplicate are re-pointed at the
    survivor first, so merging never silently drops a relationship."""
    merged = 0
    for dup_id in duplicate_ids:
        if dup_id == keep_id:
            continue
        history_store.repoint_links(dup_id, keep_id, user_id)
        history_store.delete_entry(dup_id, user_id)
        merged += 1
    return merged

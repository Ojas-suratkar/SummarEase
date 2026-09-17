"""
Auto-linked knowledge graph.

The Knowledge Base (history_store.py) already lets you search everything
you've ever summarized semantically. This goes one step further: every
time something new is saved, it's automatically compared against its
embedding-nearest past entries, and Gemini is asked *how* it relates to
each one worth mentioning -- does this new article support what an older
one said, contradict it, update/supersede it, or just touch the same
topic. The result is a real graph of relationships across your reading,
built without you doing anything -- you never manually tag or link
anything. This is deliberately the kind of connective structure a single
chat conversation can't build, because it only exists across many
separate things you've summarized over time.

Runs in the background (fire-and-forget, via
`auto_link_entry_async`) after a save -- linking is enrichment, not part
of the critical path, so a slow or failed linking pass never delays or
breaks the summary the user is actually waiting on.
"""
from __future__ import annotations

import json
import logging
import threading

import numpy as np

from . import app_context, history_store
from .embeddings import cosine_similarities
from .gemini_client import GeminiNotConfiguredError, generate_content_resilient

logger = logging.getLogger(__name__)

_TOP_CANDIDATES = 5
_SIMILARITY_THRESHOLD = 0.55
_VALID_RELATIONSHIPS = {"supports", "contradicts", "updates", "related"}

_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "relationship": {
                        "type": "string",
                        "enum": ["supports", "contradicts", "updates", "related", "unrelated"],
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One short sentence on why this relationship holds.",
                    },
                },
                "required": ["candidate_id", "relationship", "rationale"],
            },
        }
    },
    "required": ["links"],
}


class KnowledgeGraphError(RuntimeError):
    pass


def _classify_relationships(new_summary: str, candidates: list[dict]) -> list[dict]:
    candidate_lines = "\n".join(f"[{c['id']}] {c['summary'][:400]}" for c in candidates)
    prompt = (
        "A new item was just added to a personal reading knowledge base. "
        "Below is its summary, followed by a numbered list of the "
        "existing items it's most semantically similar to. For EACH "
        "candidate, classify how the new item relates to it:\n"
        "- supports: the new item backs up or reinforces the candidate's point\n"
        "- contradicts: the new item conflicts with or disputes the candidate\n"
        "- updates: the new item is a more recent/complete version of the same story or fact\n"
        "- related: same topic/theme, but not clearly supporting, contradicting, or updating\n"
        "- unrelated: not actually a meaningful connection despite the similar wording\n\n"
        "Be conservative -- most pairs that are merely topically adjacent "
        "should be 'related', and pairs that just share vocabulary but "
        "aren't meaningfully connected should be 'unrelated'.\n\n"
        f"NEW ITEM:\n{new_summary[:1500]}\n\nCANDIDATES:\n{candidate_lines}"
    )
    from google.genai import types

    response = generate_content_resilient(
        prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=_LINK_SCHEMA,
        ),
    )
    data = json.loads(response.text)
    return data.get("links", [])


def auto_link_entry(user_id: int, entry_id: int) -> int:
    """Find and record relationships between `entry_id` and its most
    similar past entries (scoped to `user_id` -- never links across
    different accounts). Returns how many links were created. Best-
    effort: any failure (no API key, bad response, no candidates) simply
    results in zero links, never an exception the caller has to handle,
    since this always runs after the thing the user actually asked for
    has already succeeded."""
    entry = history_store.get_entry_for_user(entry_id, user_id)
    if entry is None or not entry.get("embedding"):
        return 0

    candidates_raw = history_store.other_embedded_entries(user_id, entry_id)
    if not candidates_raw:
        return 0

    query_vector = np.frombuffer(entry["embedding"], dtype=np.float32)
    matrix = np.vstack(
        [np.frombuffer(c["embedding"], dtype=np.float32) for c in candidates_raw]
    )
    sims = cosine_similarities(query_vector, matrix)

    scored = sorted(zip(candidates_raw, sims), key=lambda cs: -cs[1])
    top = [
        {"id": c["id"], "summary": c["summary"], "similarity": float(sim)}
        for c, sim in scored[:_TOP_CANDIDATES]
        if sim >= _SIMILARITY_THRESHOLD
    ]
    if not top:
        return 0

    try:
        classifications = _classify_relationships(entry["summary"], top)
    except GeminiNotConfiguredError:
        return 0
    except Exception as exc:  # noqa: BLE001 -- linking is best-effort enrichment
        logger.warning("Auto-linking failed for entry %s: %s", entry_id, exc)
        return 0

    sim_by_id = {c["id"]: c["similarity"] for c in top}
    created = 0
    for link in classifications:
        relationship = link.get("relationship", "unrelated")
        candidate_id = link.get("candidate_id")
        if relationship not in _VALID_RELATIONSHIPS or candidate_id not in sim_by_id:
            continue
        try:
            history_store.add_link(
                user_id,
                entry_id,
                candidate_id,
                relationship,
                link.get("rationale", ""),
                sim_by_id[candidate_id],
            )
            created += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save link %s -> %s: %s", entry_id, candidate_id, exc)
    return created


def auto_link_entry_async(user_id: int, entry_id: int) -> None:
    """Fire-and-forget: run auto_link_entry on a daemon thread (inside a
    real Flask app context, since it needs database access -- see
    app_context.py) so it never adds latency to the request that
    triggered the save."""
    threading.Thread(
        target=lambda: app_context.run(auto_link_entry, user_id, entry_id), daemon=True
    ).start()


# ---------------------------------------------------------------------------
# SVG rendering -- a lightweight circular-layout graph, no JS charting
# library needed for a few dozen nodes and edges.
# ---------------------------------------------------------------------------

_RELATIONSHIP_COLORS = {
    "supports": "#16a34a",
    "contradicts": "#dc2626",
    "updates": "#0891b2",
    "related": "#6b7280",
}


def render_graph_svg(graph: dict, width: int = 900, height: int = 640) -> str:
    """Render `{"nodes": [...], "edges": [...]}` (see
    history_store.graph_data) as a self-contained SVG string: nodes placed
    on a circle (simple, deterministic, no physics simulation needed for
    this scale), edges as lines colored by relationship, with a small
    legend."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        return '<svg width="100%" height="120" xmlns="http://www.w3.org/2000/svg"></svg>'

    cx, cy = width / 2, (height - 60) / 2 + 20
    radius = min(cx, cy) - 70
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = (2 * 3.141592653589793 * i) / n - 3.141592653589793 / 2
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        positions[node["id"]] = (x, y)

    def esc(s: str) -> str:
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Inter, sans-serif">'
    ]

    for edge in edges:
        if edge["from_id"] not in positions or edge["to_id"] not in positions:
            continue
        x1, y1 = positions[edge["from_id"]]
        x2, y2 = positions[edge["to_id"]]
        color = _RELATIONSHIP_COLORS.get(edge["relationship"], "#9ca3af")
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="1.6" stroke-opacity="0.65" />'
        )

    for node in nodes:
        x, y = positions[node["id"]]
        label = (node.get("source_ref") or node.get("source_type") or "")[:28]
        svg_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#4f46e5" stroke="#fff" stroke-width="2">'
            f"<title>{esc(node.get('summary', ''))}</title></circle>"
        )
        text_anchor = "start" if x >= cx else "end"
        dx = 14 if x >= cx else -14
        svg_parts.append(
            f'<text x="{x + dx:.1f}" y="{y:.1f}" font-size="11" fill="#1a1d29" '
            f'text-anchor="{text_anchor}" dominant-baseline="middle">{esc(label)}</text>'
        )

    legend_y = height - 34
    legend_x = 16
    for label, color in _RELATIONSHIP_COLORS.items():
        svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3" />')
        svg_parts.append(f'<text x="{legend_x + 24}" y="{legend_y + 4}" font-size="11" fill="#5b6072">{label}</text>')
        legend_x += 110

    svg_parts.append("</svg>")
    return "".join(svg_parts)

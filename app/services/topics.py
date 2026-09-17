"""
Topic view for the Knowledge Base: groups everything you've ever
summarized into topic clusters with k-means over their embeddings
(history_store.raw_clusters -- unsupervised learning, no API call), then
asks Gemini for a short human-readable label per cluster.

This is the difference between a knowledge base that's just a flat list
you have to scroll through and one that shows you, at a glance, the
handful of things you actually keep reading about.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .gemini_client import GeminiNotConfiguredError, generate_content_resilient
from .history_store import raw_clusters


class TopicsError(RuntimeError):
    pass


def _label_cluster(summaries: list[str]) -> str:
    joined = "\n".join(f"- {s[:200]}" for s in summaries[:8])
    prompt = (
        "These are summaries of several things someone has read, all "
        "judged (by an embedding-similarity clustering algorithm) to be "
        "about the same general topic. Reply with ONLY a short topic "
        "label, 2-5 words, no punctuation at the end, nothing else.\n\n"
        f"{joined}"
    )
    response = generate_content_resilient(prompt)
    label = (response.text or "").strip().strip('"').strip("'")
    return label or "Untitled topic"


def get_topic_clusters(user_id: int, n_clusters: int = 5) -> list[dict]:
    clusters = raw_clusters(user_id, n_clusters=n_clusters)
    if not clusters:
        return []

    # Labeling clusters is N independent Gemini calls -- run them
    # concurrently rather than one after another.
    def _label_or_fallback(index_cluster: tuple[int, dict]) -> dict:
        index, cluster = index_cluster
        summaries = [item["summary"] for item in cluster["items"]]
        try:
            label = _label_cluster(summaries)
        except GeminiNotConfiguredError as exc:
            raise TopicsError(str(exc)) from exc
        except Exception:
            label = f"Topic {index + 1}"
        return {"label": label, "items": cluster["items"]}

    with ThreadPoolExecutor(max_workers=min(len(clusters), 5)) as executor:
        labeled = list(executor.map(_label_or_fallback, enumerate(clusters)))

    return labeled

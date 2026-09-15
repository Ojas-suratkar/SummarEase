"""
Generic news/article text extraction.

Design notes
------------
The original project had four separate, hand-written scrapers (NDTV, Times
of India, CNN, BBC) that pulled text out of specific CSS classes on each
site. That approach breaks the moment any of those sites redesigns a page,
and only supports four domains. This rebuild uses `trafilatura`, a
general-purpose content-extraction library that works across arbitrary news
and blog sites by analyzing page structure rather than matching hard-coded
selectors.
"""
from __future__ import annotations

import trafilatura


class ArticleExtractionError(RuntimeError):
    pass


def extract_article_text(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ArticleExtractionError("No URL provided.")

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ArticleExtractionError(f"Could not fetch the page at {url}")

    text = trafilatura.extract(downloaded, favor_recall=True)
    if not text or not text.strip():
        raise ArticleExtractionError(
            "Could not find readable article text on that page."
        )
    return text.strip()

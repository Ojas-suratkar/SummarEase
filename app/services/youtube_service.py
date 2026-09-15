"""
YouTube video summarization.

Design notes
------------
The original project ran a local BART model (via `transformers` + `torch`)
for this step, plus a headless-Selenium scrape of the comments feed for
sentiment analysis. Both add heavyweight, fragile dependencies (a multi-GB
ML runtime, and a scraper that breaks whenever YouTube's DOM or Chrome's
install path changes -- the original even hard-coded a Windows-only Chrome
binary path). This rebuild reuses the same Gemini client already used
elsewhere in the app: the transcript is chunked to stay under the model's
input limits, each chunk is summarized, and the chunk summaries are then
combined into one final summary. Comment sentiment analysis is left out
rather than shipped in a form that would silently break on most machines.
"""
from __future__ import annotations

import re

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from .summarizer import summarize_text

_VIDEO_ID_PATTERNS = (
    re.compile(r"(?:v=|/videos/|youtu\.be/|/embed/|/v/)([\w-]{11})"),
)
_WORDS_PER_CHUNK = 800


class YoutubeServiceError(RuntimeError):
    pass


def extract_video_id(url: str) -> str:
    url = (url or "").strip()
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    # Fall back to treating a bare 11-character ID as-is.
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    raise YoutubeServiceError(f"Could not find a video ID in: {url}")


def _fetch_transcript_text(video_id: str) -> str:
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        raise YoutubeServiceError(
            "This video has no captions/transcript available to summarize."
        ) from exc
    return " ".join(segment["text"] for segment in segments)


def _chunk_words(text: str, words_per_chunk: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def summarize_video(url: str, sentences: int = 6) -> dict:
    video_id = extract_video_id(url)
    transcript = _fetch_transcript_text(video_id)
    if not transcript.strip():
        raise YoutubeServiceError("Transcript was empty.")

    chunks = _chunk_words(transcript, _WORDS_PER_CHUNK)
    if len(chunks) == 1:
        final_summary = summarize_text(transcript, sentences=sentences)
    else:
        chunk_summaries = [summarize_text(chunk, sentences=3) for chunk in chunks]
        final_summary = summarize_text(
            " ".join(chunk_summaries), sentences=sentences
        )

    return {
        "video_id": video_id,
        "transcript_word_count": len(transcript.split()),
        "summary": final_summary,
    }

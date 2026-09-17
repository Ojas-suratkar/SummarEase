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
elsewhere in the app: the transcript is pre-filtered with TextRank on
long videos, chunked to stay under the model's input limits, and chunks
are summarized *concurrently* (ThreadPoolExecutor -- these are
independent I/O-bound Gemini calls, so running them in parallel instead
of one after another is a real wall-clock win on long transcripts, not
just a cosmetic change) before being combined into one final summary.
Comment sentiment analysis is left out of this module (see
youtube_comments.py, which uses the official YouTube Data API instead)
rather than shipped in a form that would silently break on most
machines.

Videos without captions
------------------------
Not every video has a captions/transcript track (auto-captions can be
disabled, or never generated for a given language), and `youtube_transcript_api`
can fail for other reasons too -- a transient network block, an
unsupported video, a library/YouTube-side API change. Rather than
failing those videos outright, `summarize_video` treats any transcript
failure the same way: it falls back to `summarizer.summarize_youtube_url`,
which hands the video URL directly to Gemini. Gemini processes the
actual audio/visual stream itself, so this works on essentially any
public video regardless of captions.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from youtube_transcript_api import YouTubeTranscriptApi

from .summarizer import DETAIL_LEVELS, summarize_text, summarize_youtube_url
from .textrank import extractive_summary

_VIDEO_ID_PATTERNS = (
    re.compile(r"(?:v=|/videos/|youtu\.be/|/embed/|/v/)([\w-]{11})"),
)
_WORDS_PER_CHUNK = 800
_TEXTRANK_THRESHOLD_WORDS = 900
_MAX_PARALLEL_CHUNKS = 4


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
    except Exception:
        # Anything that stops us getting a transcript -- captions
        # disabled, no transcript in any language, YouTube blocking the
        # request, a transient network error, a library/YouTube-side API
        # change -- is treated the same way: summarize_video() below
        # falls back to Gemini's direct video understanding instead of
        # failing the request outright.
        return ""
    return " ".join(segment["text"] for segment in segments)


def _chunk_words(text: str, words_per_chunk: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def summarize_video(url: str, detail: str = "standard", report=lambda msg: None) -> dict:
    video_id = extract_video_id(url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    settings = DETAIL_LEVELS.get(detail, DETAIL_LEVELS["standard"])
    sentences = settings["sentences"]

    report("Fetching captions...")
    transcript = _fetch_transcript_text(video_id)

    if not transcript.strip():
        report("No captions available -- asking Gemini to watch the video directly...")
        final_summary = summarize_youtube_url(canonical_url, sentences=sentences)
        return {
            "video_id": video_id,
            "transcript_word_count": 0,
            "summary": final_summary,
            "method": "video",
            "full_text": final_summary,
        }

    working_text = transcript
    word_count = len(transcript.split())
    if word_count > _TEXTRANK_THRESHOLD_WORDS:
        report(f"Long transcript ({word_count} words) -- ranking sentences with TextRank...")
        salient = extractive_summary(transcript, ratio=settings["ratio"])
        working_text = " ".join(salient) if salient else transcript

    chunks = _chunk_words(working_text, _WORDS_PER_CHUNK)
    if len(chunks) == 1:
        report("Summarizing with Gemini...")
        final_summary = summarize_text(working_text, sentences=sentences)
    else:
        report(f"Summarizing {len(chunks)} sections in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(chunks), _MAX_PARALLEL_CHUNKS)) as executor:
            chunk_summaries = list(executor.map(lambda c: summarize_text(c, sentences=3), chunks))
        report("Combining section summaries...")
        final_summary = summarize_text(" ".join(chunk_summaries), sentences=sentences)

    return {
        "video_id": video_id,
        "transcript_word_count": word_count,
        "summary": final_summary,
        "method": "transcript",
        "full_text": transcript,
    }

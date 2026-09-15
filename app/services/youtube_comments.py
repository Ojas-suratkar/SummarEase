"""
YouTube comment sentiment, via the official YouTube Data API and a
lightweight lexicon-based sentiment scorer.

Design notes
------------
The original project got comments by driving a headless Chrome browser
with Selenium to scroll and scrape YouTube's comment feed -- fragile (it
breaks whenever YouTube changes its page structure), against YouTube's
terms of service, and it hard-coded a Windows-only path to Chrome. This
version uses YouTube's own Data API (what YouTube itself offers for
exactly this), and scores sentiment with VADER, a small lexicon-based
analyzer tuned for short, informal text like comments -- instead of
loading a multi-gigabyte BERT model for the same job.
"""
from __future__ import annotations

import os

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
_analyzer = SentimentIntensityAnalyzer()


class YoutubeCommentsError(RuntimeError):
    pass


def _api_key() -> str:
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise YoutubeCommentsError(
            "YOUTUBE_API_KEY is not set. Copy .env.example to .env and add "
            "a key from https://console.cloud.google.com/apis/credentials "
            "with the YouTube Data API v3 enabled."
        )
    return api_key


def fetch_top_level_comments(video_id: str, max_results: int = 50) -> list[str]:
    """Fetch up to `max_results` top-level comments for a video."""
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(max_results, 100),
        "order": "relevance",
        "textFormat": "plainText",
        "key": _api_key(),
    }
    response = requests.get(COMMENT_THREADS_URL, params=params, timeout=15)
    if response.status_code == 403:
        raise YoutubeCommentsError(
            "YouTube API request was refused (403) -- comments may be "
            "disabled for this video, or the API key/quota is invalid."
        )
    if response.status_code != 200:
        raise YoutubeCommentsError(f"YouTube API request failed: {response.status_code}")

    data = response.json()
    comments = []
    for item in data.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append(snippet["textDisplay"])
    return comments


def score_sentiment(comments: list[str]) -> dict:
    """Average VADER compound sentiment across `comments`, plus a label."""
    if not comments:
        return {"label": "No comments", "average_score": 0.0, "sample_size": 0}

    scores = [_analyzer.polarity_scores(comment)["compound"] for comment in comments]
    average = sum(scores) / len(scores)

    if average >= 0.05:
        label = "Mostly positive"
    elif average <= -0.05:
        label = "Mostly negative"
    else:
        label = "Mixed / neutral"

    return {"label": label, "average_score": round(average, 3), "sample_size": len(comments)}


def analyze_video_comments(video_id: str, max_results: int = 50) -> dict:
    comments = fetch_top_level_comments(video_id, max_results)
    return score_sentiment(comments)

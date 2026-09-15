# SummarEase

A small Flask app that summarizes text, PDFs, YouTube videos, and news/blog
articles using Google's Gemini API, with quick keyword extraction and
text-to-speech playback for every result.

## Features

- **Text** - paste any text, get a short summary.
- **PDF** - upload a PDF, get a summary of its extracted text.
- **YouTube** - paste a video URL, get a summary built from its transcript.
- **YouTube comment sentiment** - on the YouTube results page, analyze the
  video's top comments (via the official YouTube Data API) and get an
  average sentiment score/label.
- **Article** - paste a news/blog URL, get a summary of the article body
  (extraction works across most sites, not just a fixed list of domains).
- Every summary comes with TF-IDF-ranked keywords you can click to get a
  one-line explanation, and a "Listen" button that generates audio on demand.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set GEMINI_API_KEY and FLASK_SECRET_KEY
```

## Running

```bash
python run.py
```

Then open http://localhost:5000.

## Project layout

```
app/
  __init__.py          # app factory, loads .env and config
  config.py            # Flask config from environment variables
  routes.py            # HTTP routes (one per summarizer + two small AJAX endpoints)
  services/
    summarizer.py       # Gemini API wrapper (summarize + explain-term)
    keywords.py          # TF-IDF keyword extraction
    pdf_extractor.py     # PDF -> text (PyMuPDF)
    youtube_service.py   # transcript fetch + chunked summarization
    article_extractor.py # generic article text extraction (trafilatura)
    tts_service.py       # text -> mp3 (gTTS)
    youtube_comments.py  # comment fetch (YouTube Data API) + VADER sentiment scoring
  templates/            # Jinja templates (Bootstrap 5 via CDN)
  static/                # css, js, generated audio files
run.py
```

## Notes on design choices

This app covers the same core idea as summarization tools I'd seen built by
friends, but the implementation here is my own: a from-scratch Flask app
factory + service-module structure, environment-based secrets instead of
hard-coded keys, and a single-page-per-tool flow instead of a multi-step
session-chained one. A few deliberate simplifications versus a "summarize
everything" approach:

- **Article extraction** uses `trafilatura` (general-purpose) instead of
  writing a separate scraper per news site — more sites supported, less
  code to maintain.
- **YouTube summarization** reuses the Gemini client (chunking the
  transcript) instead of also running a separate local BART model — one
  fewer multi-gigabyte ML dependency, same summarization quality.
- **YouTube comment sentiment analysis** fetches top-level comments through
  the official YouTube Data API (instead of a browser-automation scraper,
  which is fragile and against YouTube's terms of service) and scores them
  with VADER, a small lexicon-based analyzer tuned for short, informal text
  — instead of loading a multi-gigabyte BERT/transformer model for the same
  job. Needs its own `YOUTUBE_API_KEY` (see below); the button on the
  YouTube results page is hidden if a summary hasn't been generated yet.

## Environment variables

| Variable          | Required | Purpose                                   |
|--------------------|----------|--------------------------------------------|
| `GEMINI_API_KEY`   | yes      | Google Generative AI API key               |
| `FLASK_SECRET_KEY` | yes      | Signs Flask's session cookie               |
| `YOUTUBE_API_KEY`  | no       | YouTube Data API v3 key, for comment sentiment |
| `FLASK_DEBUG`      | no       | Set to `1` to enable Flask debug mode      |

Never commit your `.env` file — it's already in `.gitignore`.

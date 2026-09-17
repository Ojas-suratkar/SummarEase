# Production image -- builds the same app that runs locally, served by
# gunicorn instead of Flask's dev server. Works as-is on Render, Railway,
# Fly.io, or any host that runs a Dockerfile; set DATABASE_URL (Postgres)
# and FLASK_SECRET_KEY as environment variables/secrets on whichever host
# you pick -- nothing in this image is host-specific.
FROM python:3.11-slim

WORKDIR /app

# System deps: PyMuPDF/fpdf2 need a couple of shared libs; psycopg2-binary
# is pure-wheel so no extra build tooling is needed for it specifically.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p instance app/static/audio

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# --workers scales with CPU on most hosts; 2 threads per worker covers the
# app's background-job polling pattern (many short-lived GET polls)
# without needing a large worker count. --timeout is generous because
# some Gemini calls (video/audio understanding) legitimately take a while.
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "120"]

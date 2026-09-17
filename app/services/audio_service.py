"""
Audio ingestion -- a genuinely new input type, not another text-analysis
panel. Podcasts, voice memos, meeting recordings: none of this app's
other summarizers can touch them (PDF wants text, Article wants a URL,
even YouTube ultimately needs either a transcript or a public video URL).
This uploads the audio file straight to Gemini, which understands audio
natively -- no separate speech-to-text step, no whisper/local ASR model
to install and maintain.
"""
from __future__ import annotations

import mimetypes
import time

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient, get_client
from .summarizer import DETAIL_LEVELS

MAX_AUDIO_BYTES = 45 * 1024 * 1024  # 45MB -- generous for a voice memo or a single podcast episode
_UPLOAD_POLL_SECONDS = 1.5
_UPLOAD_POLL_TIMEOUT = 60

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"}


class AudioServiceError(RuntimeError):
    pass


def _guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "audio/mpeg"


def _upload_and_wait(file_bytes: bytes, filename: str, mime_type: str):
    import io

    from google.genai import types

    client = get_client()
    uploaded = client.files.upload(
        file=io.BytesIO(file_bytes),
        config=types.UploadFileConfig(mime_type=mime_type, display_name=filename),
    )

    waited = 0.0
    while getattr(uploaded, "state", None) is not None and str(uploaded.state) not in ("ACTIVE", "FileState.ACTIVE"):
        if waited >= _UPLOAD_POLL_TIMEOUT:
            raise AudioServiceError("The audio file is taking unusually long to process -- please try again.")
        time.sleep(_UPLOAD_POLL_SECONDS)
        waited += _UPLOAD_POLL_SECONDS
        uploaded = client.files.get(name=uploaded.name)

    return uploaded


def summarize_audio(file_bytes: bytes, filename: str, detail: str = "standard", report=lambda msg: None) -> dict:
    if len(file_bytes) > MAX_AUDIO_BYTES:
        raise AudioServiceError(
            f"That file is too large ({len(file_bytes) / 1_000_000:.1f}MB) -- "
            f"the limit here is {MAX_AUDIO_BYTES // 1_000_000}MB."
        )

    settings = DETAIL_LEVELS.get(detail, DETAIL_LEVELS["standard"])
    mime_type = _guess_mime_type(filename)

    report("Uploading audio...")
    try:
        uploaded = _upload_and_wait(file_bytes, filename, mime_type)
    except GeminiNotConfiguredError as exc:
        raise AudioServiceError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise AudioServiceError(f"Couldn't upload that audio file: {exc}") from exc

    report("Listening and summarizing...")
    from google.genai import types

    prompt = (
        f"Listen to this audio and summarize it in about {settings['sentences']} "
        "sentences. Keep the summary factual, neutral, and self-contained. If "
        "it's a conversation or meeting, note who said what only when it's "
        "clearly identifiable. Do not add any preamble -- just return the "
        "summary itself."
    )
    try:
        response = generate_content_resilient(
            types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=uploaded.uri, mime_type=mime_type)),
                    types.Part(text=prompt),
                ]
            )
        )
        summary = (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise AudioServiceError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise AudioServiceError(
            f"{exc} Audio needs Gemini directly -- there's no offline fallback for this one."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise AudioServiceError(f"Audio summarization failed: {exc}") from exc

    report("Done.")
    return {"summary": summary, "filename": filename}

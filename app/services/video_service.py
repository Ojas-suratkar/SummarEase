"""
Video ingestion -- recorded or uploaded video (screen recording, webcam
clip, a video file someone sends you), summarized by Gemini watching it
directly: the same Files-API-upload mechanism as audio_service.py, since
Gemini's video understanding covers both the visual track and the audio
track in one pass (unlike YouTube's transcript-first path, there's no
separate captions source to prefer here).

The raw file is also saved to local disk (see routes.py's /video and
services/recording_store.py) before being handed to Gemini, so a
recording is never lost just because summarization fails or the API is
briefly down -- you can always retry summarizing a saved recording
without re-recording it.
"""
from __future__ import annotations

import mimetypes
import time

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient, get_client
from .summarizer import DETAIL_LEVELS

MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200MB -- generous for a short screen/webcam recording
_UPLOAD_POLL_SECONDS = 2.0
_UPLOAD_POLL_TIMEOUT = 180  # video processing takes Gemini longer than audio

SUPPORTED_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


class VideoServiceError(RuntimeError):
    pass


def _guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "video/webm"


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
            raise VideoServiceError("The video is taking unusually long to process -- please try again.")
        time.sleep(_UPLOAD_POLL_SECONDS)
        waited += _UPLOAD_POLL_SECONDS
        uploaded = client.files.get(name=uploaded.name)

    return uploaded


def summarize_video_file(file_bytes: bytes, filename: str, detail: str = "standard", report=lambda msg: None) -> dict:
    if len(file_bytes) > MAX_VIDEO_BYTES:
        raise VideoServiceError(
            f"That file is too large ({len(file_bytes) / 1_000_000:.1f}MB) -- "
            f"the limit here is {MAX_VIDEO_BYTES // 1_000_000}MB."
        )

    settings = DETAIL_LEVELS.get(detail, DETAIL_LEVELS["standard"])
    mime_type = _guess_mime_type(filename)

    report("Uploading video...")
    try:
        uploaded = _upload_and_wait(file_bytes, filename, mime_type)
    except GeminiNotConfiguredError as exc:
        raise VideoServiceError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise VideoServiceError(f"Couldn't upload that video: {exc}") from exc

    report("Watching and summarizing...")
    from google.genai import types

    prompt = (
        f"Watch this video and summarize it in about {settings['sentences']} "
        "sentences, covering both what's shown and what's said. Keep it "
        "factual, neutral, and self-contained. Do not add any preamble -- "
        "just return the summary itself."
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
        raise VideoServiceError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise VideoServiceError(
            f"{exc} Video needs Gemini directly -- there's no offline fallback for this one."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise VideoServiceError(f"Video summarization failed: {exc}") from exc

    report("Done.")
    return {"summary": summary, "filename": filename}

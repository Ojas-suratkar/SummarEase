"""
Image/screenshot understanding -- another genuinely new input type.
Infographics, whiteboard photos, receipts, charts, scanned pages: none of
this app's text-based extractors (trafilatura, PyMuPDF) can read a photo.
Sent inline to Gemini's vision input (small enough not to need the Files
API upload flow audio_service.py uses) with a task-aware prompt so a
receipt gets itemized rather than vaguely "summarized."
"""
from __future__ import annotations

import mimetypes

from .gemini_client import GeminiNotConfiguredError, GeminiUnavailableError, generate_content_resilient

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB -- comfortably inline, no Files API needed
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

ANALYSIS_MODES = {
    "describe": "Describe what's in this image in detail -- the overall subject, key visual elements, and any text visible in it.",
    "extract_text": "Extract every piece of text visible in this image, preserving its structure (headings, lists, table rows) as best you can. Return only the extracted text.",
    "receipt": "This is a receipt or invoice. Extract each line item with its price, plus the subtotal, tax, and total if visible. Return it as a clean, readable itemized list.",
    "chart": "This is a chart, graph, or infographic. Explain what it shows: the axes/categories, the key trend or comparison, and the most important takeaway a reader should walk away with.",
    "whiteboard": "This is a photo of a whiteboard or handwritten notes. Transcribe the content as cleanly as possible, organizing it into a structured summary (headings, bullet points) rather than a wall of text.",
}


class ImageServiceError(RuntimeError):
    pass


def _guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "image/png"


def analyze_image(file_bytes: bytes, filename: str, mode: str = "describe") -> dict:
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ImageServiceError(
            f"That image is too large ({len(file_bytes) / 1_000_000:.1f}MB) -- "
            f"the limit here is {MAX_IMAGE_BYTES // 1_000_000}MB."
        )
    if not file_bytes:
        raise ImageServiceError("Please choose an image file.")

    prompt = ANALYSIS_MODES.get(mode, ANALYSIS_MODES["describe"])
    mime_type = _guess_mime_type(filename)

    from google.genai import types

    try:
        response = generate_content_resilient(
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=file_bytes, mime_type=mime_type)),
                    types.Part(text=prompt),
                ]
            )
        )
        result_text = (response.text or "").strip()
    except GeminiNotConfiguredError as exc:
        raise ImageServiceError(str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise ImageServiceError(
            f"{exc} Image analysis needs Gemini directly -- there's no offline fallback for this one."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ImageServiceError(f"Image analysis failed: {exc}") from exc

    return {"result": result_text, "mode": mode, "filename": filename}

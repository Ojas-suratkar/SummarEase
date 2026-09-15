"""
Text-to-speech, using gTTS, saved into the app's own static/audio folder.

Design notes
------------
The original hard-coded a Windows path (`D:/SummarEase/static/output.mp3`),
which only worked on the original author's machine. This version resolves
the output path relative to the Flask app's static folder, so it works the
same way on Windows, macOS, and Linux, and gives each request its own
filename instead of overwriting a single shared file (which would race
under concurrent requests).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from gtts import gTTS

AUDIO_SUBDIR = "audio"


def synthesize(text: str, static_folder: str) -> str:
    """Generate an mp3 for `text` and return its filename (not full path)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No text to synthesize.")

    audio_dir = Path(static_folder) / AUDIO_SUBDIR
    audio_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = audio_dir / filename

    # gTTS has a practical input-length limit; keep requests reasonable.
    gTTS(text=text[:5000]).save(str(output_path))
    return filename

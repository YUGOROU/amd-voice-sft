"""
Whisper STT wrapper — ROCm-compatible.

Two modes:
  1. Local Whisper (runs on the same MI300X, via openai-whisper or faster-whisper)
  2. Remote endpoint (OpenAI-compatible /v1/audio/transcriptions)

Set STT_MODE env var to "local" or "remote". Defaults to "remote" in HF Space.
"""

import os
from pathlib import Path

STT_MODE     = os.getenv("STT_MODE", "remote")
STT_URL      = os.getenv("STT_URL", "")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")   # tiny/base/small/medium


# ---------------------------------------------------------------------------
# Remote path (OpenAI-compatible endpoint — used in HF Space)
# ---------------------------------------------------------------------------

def _transcribe_remote(audio_path: str) -> str:
    import httpx
    url = f"{STT_URL.rstrip('/')}/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        r = httpx.post(
            url,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"model": "whisper-1"},
            timeout=30.0,
        )
    r.raise_for_status()
    return r.json()["text"].strip()


# ---------------------------------------------------------------------------
# Local path (runs on MI300X with ROCm)
# ---------------------------------------------------------------------------

_whisper_model = None

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(WHISPER_SIZE)
    return _whisper_model


def _transcribe_local(audio_path: str) -> str:
    model = _load_whisper()
    result = model.transcribe(audio_path, fp16=True, language="en")
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio_path: str) -> str:
    """Transcribe audio file to text. Returns empty string on failure."""
    if not audio_path or not Path(audio_path).exists():
        return ""
    try:
        if STT_MODE == "local":
            return _transcribe_local(audio_path)
        return _transcribe_remote(audio_path)
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""

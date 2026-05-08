"""
TTS wrapper — Piper (primary) or Coqui (fallback).

The latency-hiding trick: the structured output parser fires TTS on the opening
line BEFORE the think block is processed, so users hear a response in <1.5s.

Two modes:
  1. Remote endpoint (OpenAI-compatible /v1/audio/speech) — used in HF Space
  2. Local Piper subprocess — used on MI300X for lowest latency

Set TTS_MODE env var to "local" or "remote".
"""

import os
import subprocess
import tempfile
from pathlib import Path

TTS_MODE   = os.getenv("TTS_MODE", "remote")
TTS_URL    = os.getenv("TTS_URL", "")
PIPER_BIN  = os.getenv("PIPER_BIN", "piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")


# ---------------------------------------------------------------------------
# Remote path (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def _synthesize_remote(text: str) -> str | None:
    import httpx
    try:
        r = httpx.post(
            f"{TTS_URL.rstrip('/')}/v1/audio/speech",
            json={"model": "tts-1", "input": text, "response_format": "wav"},
            timeout=30.0,
        )
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"[TTS] Remote error: {e}")
        return None


# ---------------------------------------------------------------------------
# Local Piper path (MI300X)
# ---------------------------------------------------------------------------

def _synthesize_local(text: str) -> str | None:
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", tmp.name],
            input=text.encode(),
            check=True,
            capture_output=True,
        )
        return tmp.name
    except Exception as e:
        print(f"[TTS] Local Piper error: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(text: str) -> str | None:
    """Convert text to speech. Returns path to WAV file or None on failure."""
    text = text.strip()
    if not text:
        return None
    if TTS_MODE == "local":
        return _synthesize_local(text)
    if TTS_URL:
        return _synthesize_remote(text)
    return None

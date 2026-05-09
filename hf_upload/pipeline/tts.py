"""
TTS wrapper — edge-tts (remote, OpenAI-compatible endpoint) or local Piper.

The endpoint accepts a `voice` field to select per-profile voices.
"""

import os
import subprocess
import tempfile

TTS_MODE    = os.getenv("TTS_MODE", "remote")
TTS_URL     = os.getenv("TTS_URL", "")
PIPER_BIN   = os.getenv("PIPER_BIN", "piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")


def _synthesize_remote(text: str, voice: str, rate: str, pitch: str) -> str | None:
    import httpx
    try:
        r = httpx.post(
            f"{TTS_URL.rstrip('/')}/v1/audio/speech",
            json={
                "model":           "tts-1",
                "input":           text,
                "voice":           voice,
                "response_format": "mp3",
                "rate":            rate,
                "pitch":           pitch,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"[TTS] Remote error: {e}")
        return None


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


def synthesize(
    text:    str,
    voice:   str = "en-US-JennyNeural",
    rate:    str = "-5%",
    pitch:   str = "-3Hz",
) -> str | None:
    """Convert text to speech. Returns path to MP3/WAV file or None on failure."""
    text = text.strip()
    if not text:
        return None
    if TTS_MODE == "local":
        return _synthesize_local(text)
    if TTS_URL:
        return _synthesize_remote(text, voice, rate, pitch)
    return None

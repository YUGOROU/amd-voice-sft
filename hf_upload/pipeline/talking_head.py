"""
SadTalker client — sends portrait + audio to the MI300X server, gets MP4 back.

Server (droplet port 8003):
  POST /generate  multipart: audio file + profile_id form field → MP4 bytes
  GET  /health    → {"status":"ok"}
"""

import os
import tempfile
import httpx

TALKING_HEAD_URL = os.getenv("TALKING_HEAD_URL", "")
TIMEOUT          = float(os.getenv("TALKING_HEAD_TIMEOUT", "120"))


def generate_talking_video(audio_path: str, profile_id: str) -> str | None:
    """
    Animate the profile portrait to match the given audio.
    Returns path to a temp MP4 file, or None if unavailable/failed.
    """
    if not TALKING_HEAD_URL or not audio_path:
        return None
    if not os.path.exists(audio_path):
        return None
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        suffix  = os.path.splitext(audio_path)[1] or ".mp3"
        mime    = "audio/mpeg" if suffix == ".mp3" else "audio/wav"

        r = httpx.post(
            f"{TALKING_HEAD_URL.rstrip('/')}/generate",
            files={"audio": (f"audio{suffix}", audio_bytes, mime)},
            data={"profile_id": profile_id},
            timeout=TIMEOUT,
        )
        r.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name

    except Exception as e:
        print(f"[TalkingHead] {e}")
        return None


def is_available() -> bool:
    if not TALKING_HEAD_URL:
        return False
    try:
        r = httpx.get(f"{TALKING_HEAD_URL.rstrip('/')}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False

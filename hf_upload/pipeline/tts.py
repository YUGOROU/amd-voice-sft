"""
TTS — Kokoro ONNX (primary) with local edge-tts (secondary/accents).
Runs locally on the HF Space for zero-latency and reliable voice updates.
"""

import os
import asyncio
import tempfile
import threading

# ---------------------------------------------------------------------------
# Kokoro ONNX (High Quality Neural)
# ---------------------------------------------------------------------------

_kokoro = None

def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        try:
            from kokoro_onnx import Kokoro
            from huggingface_hub import hf_hub_download
            onnx_path    = hf_hub_download("thewh1teagle/kokoro-onnx", "kokoro-v0_19.onnx")
            voices_path  = hf_hub_download("thewh1teagle/kokoro-onnx", "voices-v0_19.bin")
            _kokoro = Kokoro(onnx_path, voices_path)
            print("[TTS] Kokoro ONNX loaded")
        except Exception as e:
            print(f"[TTS] Kokoro init failed: {e}")
    return _kokoro


def _synthesize_kokoro(text: str, voice: str, speed: float) -> str | None:
    model = _get_kokoro()
    if model is None:
        return None
    try:
        import soundfile as sf
        samples, sr = model.create(text, voice=voice, speed=speed, lang="en-us")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, samples, sr)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"[TTS] Kokoro synthesis error: {e}")
        return None


# ---------------------------------------------------------------------------
# Local Edge-TTS (Diverse Accents / Fallback)
# ---------------------------------------------------------------------------

def _run_edge_tts(text: str, voice: str, rate: str, pitch: str, output_path: str):
    """Internal helper to run the async edge-tts communicate."""
    import edge_tts
    async def _amain():
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)
    
    # Run in a new event loop since we're in a threaded environment (Gradio/FastAPI)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_amain())
    finally:
        loop.close()


def _synthesize_edge_local(text: str, voice: str, rate: str, pitch: str) -> str | None:
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_name = tmp.name
        tmp.close()
        
        # edge-tts is async, but Gradio is sync-threaded. Run it in a helper.
        _run_edge_tts(text, voice, rate, pitch, tmp_name)
        return tmp_name
    except Exception as e:
        print(f"[TTS] Local Edge-TTS error: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(
    text:         str,
    kokoro_voice: str | None = "af_sky",
    kokoro_speed: float      = 1.0,
    edge_voice:   str        = "en-US-JennyNeural",
    rate:         str        = "+0%",
    pitch:        str        = "+0Hz",
) -> str | None:
    """Return path to synthesized audio file, or None on failure."""
    text = text.strip()
    if not text:
        return None

    # 1. Try Kokoro if specified
    if kokoro_voice:
        result = _synthesize_kokoro(text, kokoro_voice, kokoro_speed)
        if result:
            return result

    # 2. Otherwise use local Edge-TTS (direct, no remote server)
    return _synthesize_edge_local(text, edge_voice, rate, pitch)

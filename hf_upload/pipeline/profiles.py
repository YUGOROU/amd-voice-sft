"""
Avatar profiles.

kokoro_voice  — Kokoro ONNX voice ID (primary, high quality)
kokoro_speed  — playback speed for Kokoro (1.0 = normal)
edge_voice    — edge-tts voice name (fallback or for accents Kokoro lacks)
rate / pitch  — edge-tts rate / pitch tweaks (fallback only)
"""

PROFILES: dict[str, dict] = {

    # ── React frontend avatars ────────────────────────────────────────────

    "lumi": {
        "id":           "lumi",
        "display_name": "Lumi",
        "label":        "Lumi · Your Guide",
        "portrait":     "lumi.png",
        "kokoro_voice": "af_sky",
        "kokoro_speed": 1.1,
        "edge_voice":   "en-US-JennyNeural",
        "rate":         "+5%",
        "pitch":        "+4Hz",
        "description":  "a bright, cheerful, and enthusiastic AI guide with a touch of anime-style energy",
        "voice": "af_sky", "speed": 1.1,
    },

    "dorothy": {
        "id":           "dorothy",
        "display_name": "Dorothy",
        "label":        "Dorothy · Very Elderly Black woman, 80",
        "portrait":     "dorothy.png",
        # Michelle (US) + Slow rate = Grandmotherly feel.
        "kokoro_voice": None,
        "kokoro_speed": 1.0,
        "edge_voice":   "en-US-MichelleNeural",
        "rate":         "-15%",
        "pitch":        "-3Hz",
        "description":  "a very wise, patient, and nurturing elderly woman with a slow, warm, and grandmotherly heart",
        "voice": "en-US-MichelleNeural", "speed": 1.0,
    },
 
    "harold": {
        "id":           "harold",
        "display_name": "Harold",
        "label":        "Harold · Very Elderly White man, 80",
        "portrait":     "harold.png",
        # Roger (US) is the most realistic "Grandfather" voice.
        "kokoro_voice": None,
        "kokoro_speed": 1.0,
        "edge_voice":   "en-US-RogerNeural",
        "rate":         "-15%",
        "pitch":        "-5Hz",
        "description":  "a very gentle, patient, and distinguished elderly gentleman with a slow, wise, and deep voice",
        "voice": "en-US-RogerNeural", "speed": 1.0,
    },

    "marcus": {
        "id":           "marcus",
        "display_name": "Marcus",
        "label":        "Marcus · Young Black man, 20",
        "portrait":     "marcus.png",
        # Steffan (US) + 10% rate = high energy Gen Z vibe.
        "kokoro_voice": None,
        "kokoro_speed": 1.0,
        "edge_voice":   "en-US-SteffanNeural",
        "rate":         "+10%",
        "pitch":        "+2Hz",
        "description":  "a warm, energetic Gen Z young man who is upbeat, uses a bit of modern friendly energy, and is very encouraging",
        "voice": "en-US-SteffanNeural", "speed": 1.0,
    },

    "priya": {
        "id":           "priya",
        "display_name": "Priya",
        "label":        "Priya · Young South Asian woman, 28",
        "portrait":     "priya.png",
        "kokoro_voice": None,
        "kokoro_speed": 1.0,
        "edge_voice":   "en-IN-NeerjaNeural",
        "rate":         "+3%",
        "pitch":        "+2Hz",
        "description":  "a warm, attentive, and bright young South Asian woman",
        "voice": "en-IN-NeerjaNeural", "speed": 1.0,
    },

    "carlos": {
        "id":           "carlos",
        "display_name": "Carlos",
        "label":        "Carlos · Middle-aged Latino man, 45",
        "portrait":     "carlos.png",
        "kokoro_voice": None,
        "kokoro_speed": 1.0,
        "edge_voice":   "en-US-ChristopherNeural",
        "rate":         "-2%",
        "pitch":        "-3Hz",
        "description":  "a knowledgeable, confident, and experienced advisor and mentor",
        "voice": "en-US-ChristopherNeural", "speed": 1.0,
    },

    # ── Additional Gradio UI profiles ─────────────────────────────────────

    "sophie": {
        "id": "sophie", "display_name": "Sophie",
        "label": "Sophie · Young White woman, 25",
        "portrait": "sophie.png",
        "kokoro_voice": "af_sarah", "kokoro_speed": 1.05,
        "edge_voice": "en-US-AriaNeural", "rate": "+3%", "pitch": "+3Hz",
        "description": "a bright, caring, and cheerful young woman",
        "voice": "af_sarah", "speed": 1.05,
    },
    "james": {
        "id": "james", "display_name": "James",
        "label": "James · Middle-aged White man, 45",
        "portrait": "james.png",
        "kokoro_voice": None, "kokoro_speed": 1.0,
        "edge_voice": "en-US-GuyNeural", "rate": "-2%", "pitch": "-3Hz",
        "description": "a calm, thoughtful, and reassuring man in his 40s",
        "voice": "en-US-GuyNeural", "speed": 1.0,
    },
    "mei": {
        "id": "mei", "display_name": "Mei",
        "label": "Mei · Young East Asian woman, 25",
        "portrait": "mei.png",
        "kokoro_voice": "af_nicole", "kokoro_speed": 1.05,
        "edge_voice": "en-US-JennyNeural", "rate": "+3%", "pitch": "+2Hz",
        "description": "a kind, gentle, and perceptive young woman",
        "voice": "af_nicole", "speed": 1.05,
    },
    "amara": {
        "id": "amara", "display_name": "Amara",
        "label": "Amara · Middle-aged Black woman, 45",
        "portrait": "amara.png",
        "kokoro_voice": "af_bella", "kokoro_speed": 0.95,
        "edge_voice": "en-US-MonicaNeural", "rate": "-3%", "pitch": "-2Hz",
        "description": "a strong, empathetic, and grounded woman",
        "voice": "af_bella", "speed": 0.95,
    },
    "liam": {
        "id": "liam", "display_name": "Liam",
        "label": "Liam · Young Irish man, 28",
        "portrait": "liam.png",
        "kokoro_voice": None, "kokoro_speed": 1.0,
        "edge_voice": "en-IE-ConnorNeural", "rate": "+2%", "pitch": "+0Hz",
        "description": "a cheerful, witty, and dependable young man",
        "voice": "en-IE-ConnorNeural", "speed": 1.0,
    },
}

DEFAULT_PROFILE_ID = "sophie"


def get_profile(profile_id: str) -> dict:
    return PROFILES.get(profile_id, PROFILES[DEFAULT_PROFILE_ID])


def profile_labels() -> list[str]:
    return [p["label"] for pid, p in PROFILES.items() if not pid.endswith("_gradio")]


def id_from_label(label: str) -> str:
    for pid, p in PROFILES.items():
        if p["label"] == label:
            return pid
    return DEFAULT_PROFILE_ID

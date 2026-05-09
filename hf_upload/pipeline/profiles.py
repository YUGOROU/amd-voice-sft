"""
Avatar profiles — each defines a portrait image, edge-tts voice, and rate/pitch.

Portrait images live in hf_upload/portraits/<profile_id>.jpg
They must be front-facing, good lighting, neutral expression (~512×512+).
"""

PROFILES: dict[str, dict] = {
    "marcus": {
        "id":           "marcus",
        "display_name": "Marcus",
        "label":        "Marcus · Young Black man, 25",
        "portrait":     "marcus.png",
        "voice":        "en-US-DavisNeural",
        "rate":         "+5%",
        "pitch":        "-2Hz",
        "description":  "Young, warm, and energetic",
    },
    "sophie": {
        "id":           "sophie",
        "display_name": "Sophie",
        "label":        "Sophie · Young White woman, 25",
        "portrait":     "sophie.jpg",
        "voice":        "en-US-AriaNeural",
        "rate":         "+3%",
        "pitch":        "+3Hz",
        "description":  "Bright, caring, and cheerful",
    },
    "james": {
        "id":           "james",
        "display_name": "James",
        "label":        "James · Middle-aged White man, 45",
        "portrait":     "james.jpg",
        "voice":        "en-US-ChristopherNeural",
        "rate":         "-2%",
        "pitch":        "-2Hz",
        "description":  "Calm, thoughtful, and reassuring",
    },
    "dorothy": {
        "id":           "dorothy",
        "display_name": "Dorothy",
        "label":        "Dorothy · Elderly Black woman, 70",
        "portrait":     "dorothy.jpg",
        "voice":        "en-US-MichelleNeural",
        "rate":         "-12%",
        "pitch":        "-4Hz",
        "description":  "Wise, nurturing, and deeply caring",
    },
    "harold": {
        "id":           "harold",
        "display_name": "Harold",
        "label":        "Harold · Elderly White man, 70",
        "portrait":     "harold.jpg",
        "voice":        "en-US-RogerNeural",
        "rate":         "-15%",
        "pitch":        "-6Hz",
        "description":  "Gentle, patient, and experienced",
    },
    "priya": {
        "id":           "priya",
        "display_name": "Priya",
        "label":        "Priya · Young South Asian woman, 28",
        "portrait":     "priya.jpg",
        "voice":        "en-IN-NeerjaNeural",
        "rate":         "+2%",
        "pitch":        "+2Hz",
        "description":  "Warm, attentive, and thoughtful",
    },
    "carlos": {
        "id":           "carlos",
        "display_name": "Carlos",
        "label":        "Carlos · Middle-aged Latino man, 45",
        "portrait":     "carlos.jpg",
        "voice":        "en-US-TonyNeural",
        "rate":         "+0%",
        "pitch":        "-1Hz",
        "description":  "Friendly, supportive, and encouraging",
    },
    "mei": {
        "id":           "mei",
        "display_name": "Mei",
        "label":        "Mei · Young East Asian woman, 25",
        "portrait":     "mei.jpg",
        "voice":        "en-US-JennyNeural",
        "rate":         "+3%",
        "pitch":        "+2Hz",
        "description":  "Kind, gentle, and perceptive",
    },
    "amara": {
        "id":           "amara",
        "display_name": "Amara",
        "label":        "Amara · Middle-aged Black woman, 45",
        "portrait":     "amara.jpg",
        "voice":        "en-US-MonicaNeural",
        "rate":         "-3%",
        "pitch":        "-2Hz",
        "description":  "Strong, empathetic, and grounded",
    },
    "liam": {
        "id":           "liam",
        "display_name": "Liam",
        "label":        "Liam · Young Irish man, 28",
        "portrait":     "liam.jpg",
        "voice":        "en-IE-ConnorNeural",
        "rate":         "+2%",
        "pitch":        "+0Hz",
        "description":  "Cheerful, witty, and dependable",
    },
}

DEFAULT_PROFILE_ID = "sophie"


def get_profile(profile_id: str) -> dict:
    return PROFILES.get(profile_id, PROFILES[DEFAULT_PROFILE_ID])


def profile_labels() -> list[str]:
    return [p["label"] for p in PROFILES.values()]


def id_from_label(label: str) -> str:
    for pid, p in PROFILES.items():
        if p["label"] == label:
            return pid
    return DEFAULT_PROFILE_ID

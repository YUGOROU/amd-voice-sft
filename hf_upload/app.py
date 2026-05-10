"""
Lumi — AI Voice Companion for Dementia & Alzheimer's patients.
HuggingFace Spaces Gradio app.

Environment variables (set as HF Space Secrets):
  VLLM_BASE_URL       — vLLM OpenAI-compatible endpoint
  MODEL_NAME          — model ID served by vLLM
  STT_URL             — Whisper-compatible STT endpoint
  TTS_URL             — TTS endpoint (/v1/audio/speech)
  TALKING_HEAD_URL    — SadTalker server endpoint (port 8003)
  PATIENT_ID          — persistent ID for ChromaDB memory
  PATIENT_NAME        — patient's first name shown in the UI
"""

import os
import base64
import time
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

import gradio as gr
from openai import OpenAI

from pipeline.memory       import build_system_prompt, get_context, save_session
from pipeline.parser       import extract_facts_from_response, parse_structured_output
from pipeline.profiles     import DEFAULT_PROFILE_ID, profile_labels, id_from_label, get_profile
from pipeline.scam_filter  import check_and_deflect
from pipeline.stt          import transcribe
from pipeline.talking_head import generate_talking_video, is_available as talking_head_available
from pipeline.tts          import synthesize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "Debdeep30/lumi-qwen3-4b-grpo")
PATIENT_ID    = os.getenv("PATIENT_ID",   "demo_user_001")
PATIENT_NAME  = os.getenv("PATIENT_NAME", "Margaret")
STT_URL       = os.getenv("STT_URL",      "")
TTS_URL       = os.getenv("TTS_URL",      "")

llm = OpenAI(base_url=VLLM_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "not-required"))

PORTRAIT_DIR = Path(__file__).parent / "portraits"

def _load_portrait(profile_id: str) -> Image.Image | None:
    profile = get_profile(profile_id)
    path    = PORTRAIT_DIR / profile["portrait"]
    if path.exists():
        return Image.open(path)
    return None


# Fallback avatar images (for text chat / non-video mode)
AVATAR_DIR = Path(__file__).parent / "avatar"
AVATAR_IMAGES = {
    tag: Image.open(AVATAR_DIR / f"avatar_{tag}.png")
    for tag in ("smile", "nod", "concerned", "gentle", "laugh")
}

# VAD tuning (streaming mode — kept for reference)
SILENCE_THRESHOLD = 0.015
SILENCE_CHUNKS    = 6
MIN_SPEECH_CHUNKS = 3

# ---------------------------------------------------------------------------
# Core LLM helpers
# ---------------------------------------------------------------------------

def _get_system_prompt() -> str:
    return build_system_prompt(PATIENT_ID, PATIENT_NAME)


def _call_llm(messages: list[dict]) -> str:
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        extra_body={"repetition_penalty": 1.2},
    )
    return resp.choices[0].message.content or ""


def _call_llm_stream(messages: list[dict]):
    for chunk in llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        stream=True,
        extra_body={"repetition_penalty": 1.2},
    ):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _build_messages(
    history:        list[dict],
    companion_name: str = "Lumi",
    companion_desc: str = "a warm and patient AI companion",
) -> list[dict]:
    prompt = build_system_prompt(
        PATIENT_ID, PATIENT_NAME,
        companion_name=companion_name,
        companion_desc=companion_desc,
    )
    return [{"role": "system", "content": prompt}] + history


def _tts_and_video(text: str, profile_id: str) -> tuple[str | None, str | None]:
    """Run TTS and optionally SadTalker. Returns (audio_path, video_path)."""
    profile    = get_profile(profile_id)
    audio_path = synthesize(
        text,
        kokoro_voice=profile.get("kokoro_voice"),
        kokoro_speed=profile.get("kokoro_speed", 1.0),
        edge_voice=profile.get("edge_voice", "en-US-JennyNeural"),
        rate=profile.get("rate", "+0%"),
        pitch=profile.get("pitch", "+0Hz"),
    )
    video_path = None
    if audio_path and talking_head_available():
        video_path = generate_talking_video(audio_path, profile_id)
    return audio_path, video_path


def _audio_b64(path: str | None) -> str:
    """Read audio file, return data-URI string, then delete the temp file."""
    if not path:
        return ""
    try:
        mime = "audio/wav" if path.endswith(".wav") else "audio/mpeg"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        try:
            os.unlink(path)
        except OSError:
            pass
        return f"data:{mime};base64," + data
    except Exception as e:
        print(f"[audio_b64] {e}")
        return ""


def _end_of_session_summary(history: list[dict]) -> str:
    if len(history) < 2:
        return ""
    summary_prompt = [
        {"role": "system", "content": (
            "You are a clinical note writer. Summarise the following conversation "
            "between an elderly patient and their AI companion Lumi. "
            "Include: mood progression, key topics, any confusion noted, "
            "scam alerts (if any), memorable facts mentioned. "
            "Format: plain text, 5-8 bullet points. Be concise and factual."
        )},
        {"role": "user", "content": "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )},
    ]
    try:
        return _call_llm(summary_prompt)
    except Exception:
        return "(Summary unavailable — LLM endpoint not reachable.)"


# ---------------------------------------------------------------------------
# Gradio handlers — Text Chat
# ---------------------------------------------------------------------------

def text_chat(message: str, history: list[dict], profile_id: str):
    profile = get_profile(profile_id)
    is_scam, deflection = check_and_deflect(message)
    if is_scam:
        new_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": deflection},
        ]
        yield new_history, AVATAR_IMAGES["gentle"]
        return

    messages = _build_messages(history, profile["display_name"], profile["description"]) + [{"role": "user", "content": message}]
    partial  = ""
    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]
    for token in _call_llm_stream(messages):
        partial += token
        parsed  = parse_structured_output(partial)
        new_history[-1]["content"] = parsed["full_response"]
        yield new_history, AVATAR_IMAGES.get(parsed["avatar_tag"], AVATAR_IMAGES["smile"])

    facts = extract_facts_from_response(partial)
    save_session(PATIENT_ID, facts or [], "unknown", "unknown",
                 f"Text session — {len(new_history)//2} turns",
                 new_history)


# ---------------------------------------------------------------------------
# Gradio handlers — Voice Chat (record → submit)
# ---------------------------------------------------------------------------

def _save_wav_np(audio_tuple) -> str | None:
    """Save (sr, numpy_array) tuple to a temp WAV file. Handles float32 and int16."""
    if audio_tuple is None:
        return None
    sr, audio = audio_tuple
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    # Gradio 5.x may give float32 [-1,1] or int16 — normalise to float32 for soundfile
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    sf.write(tmp.name, audio, int(sr))
    tmp.close()
    return tmp.name


def lumi_api(message: str, history: list[dict], profile_id: str, session_id: str | None = None):
    """
    Unified API for the React frontend.
    Returns: (response_text, audio_b64_data_uri, avatar_tag, final_session_id)
    """
    profile = get_profile(profile_id)
    cname   = profile["display_name"]
    cdesc   = profile["description"]

    is_scam, deflection = check_and_deflect(message)
    if is_scam:
        audio_path, _ = _tts_and_video(deflection, profile_id)
        return deflection, _audio_b64(audio_path), "gentle", session_id

    messages = _build_messages(history, cname, cdesc) + [{"role": "user", "content": message}]
    try:
        raw = _call_llm(messages)
        parsed = parse_structured_output(raw)
        response_text = parsed["full_response"]
        avatar_tag = parsed["avatar_tag"]
        audio_path, _ = _tts_and_video(response_text, profile_id)
        
        # Real-time saving
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text}
        ]
        facts = extract_facts_from_response(raw)
        final_id = save_session(PATIENT_ID, facts or [], "unknown", "unknown", 
                                f"Chat - {len(new_history)//2} turns", 
                                new_history, session_id)

        return response_text, _audio_b64(audio_path), avatar_tag, final_id, parsed.get("action")
    except Exception as e:
        import traceback
        print(f"[API Error] {e}")
        traceback.print_exc()
        err_msg = "I'm sorry, I'm having a little trouble. Could you try again?"
        audio_path, _ = _tts_and_video(err_msg, profile_id)
        return err_msg, _audio_b64(audio_path), "concerned", session_id, None


def voice_submit(audio_tuple, history: list[dict], profile_id: str):
    """
    Called when the user finishes recording and clicks Send (or stops recording).
    Returns: (chatbot, audio_out, video_out, avatar_img, status)
    """
    no_change = (history, None, None, AVATAR_IMAGES["smile"], "🎤 Record then click Send")

    if audio_tuple is None:
        return no_change

    # STT
    wav_path = _save_wav_np(audio_tuple)
    user_text = ""
    try:
        user_text = transcribe(wav_path) if STT_URL else "[STT not configured — set STT_URL secret]"
    except Exception as e:
        print(f"[STT] {e}")
    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    if not user_text:
        return history, None, None, AVATAR_IMAGES["smile"], "❓ Couldn't hear that — try again"

    # Scam check
    is_scam, deflection = check_and_deflect(user_text)
    if is_scam:
        audio_path, video_path = _tts_and_video(deflection, profile_id)
        new_history = history + [
            {"role": "user",      "content": f"🎤 {user_text}"},
            {"role": "assistant", "content": deflection},
        ]
        return new_history, audio_path, video_path, AVATAR_IMAGES["gentle"], "🎤 Record then click Send"

    # LLM + TTS + (optionally) SadTalker
    messages = _build_messages(history) + [{"role": "user", "content": user_text}]
    response_text = ""
    audio_path    = None
    video_path    = None
    avatar        = AVATAR_IMAGES["smile"]

    try:
        raw           = _call_llm(messages)
        parsed        = parse_structured_output(raw)
        response_text = parsed["full_response"]
        avatar        = AVATAR_IMAGES.get(parsed["avatar_tag"], AVATAR_IMAGES["smile"])
        audio_path, video_path = _tts_and_video(response_text, profile_id)

        new_history = history + [
            {"role": "user",      "content": f"🎤 {user_text}"},
            {"role": "assistant", "content": response_text},
        ]

        facts = extract_facts_from_response(raw)
        save_session(PATIENT_ID, facts or [], "unknown", "unknown", 
                     "Voice session", new_history)

        return new_history, audio_path, video_path, avatar, "🎤 Record then click Send"

    except Exception as e:
        print(f"[LLM/TTS] {e}")
        response_text = "I'm sorry, I had a little trouble. Could you try again?"
        audio_path, video_path = _tts_and_video(response_text, profile_id)
        new_history = history + [{"role": "assistant", "content": response_text}]
        return new_history, audio_path, video_path, avatar, "🎤 Record then click Send"


# ---------------------------------------------------------------------------
# Profile change handlers
# ---------------------------------------------------------------------------

def change_profile(label: str):
    profile_id = id_from_label(label)
    profile    = get_profile(profile_id)
    portrait   = _load_portrait(profile_id)
    desc       = f"**{profile['display_name']}** — {profile['description']}"
    return profile_id, portrait or AVATAR_IMAGES["smile"], desc


# ---------------------------------------------------------------------------
# Family dashboard
# ---------------------------------------------------------------------------

def end_session(history: list[dict]):
    if not history:
        return "No conversation to summarise."
    summary = _end_of_session_summary(history)
    date    = time.strftime("%B %d, %Y")
    return (
        f"SESSION SUMMARY — {date}\n"
        f"Patient: {PATIENT_NAME}\n"
        f"Duration: {len(history)//2} turns\n\n"
        f"{summary}"
    )


def load_memory_display():
    ctx          = get_context(PATIENT_ID)
    facts_txt    = "\n".join(f"• {f}" for f in ctx["facts"]) or "(none yet)"
    sessions_txt = "\n\n---\n\n".join(ctx["summaries"]) or "(no previous sessions)"
    return facts_txt, sessions_txt


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
/* ================================================================
   Lumi — Clear Sky Theme (white + soft sky-blue)
   Large fonts, generous spacing, calming palette for elderly users
   ================================================================ */

/* Hide empty video drop-zone (Gradio renders it even when visible=False) */
#avatar-video .empty.svelte-p5msvs,
#avatar-video .upload,
#avatar-video .wrap.svelte-1dmqpg4,
.video-component .empty { display: none !important; }

/* Base */
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    background: #f7fbff !important;
    font-family: 'Georgia', serif !important;
}

/* Soft sky-blue page background */
body, .main, footer {
    background: #f0f6ff !important;
}

/* ── Header ─────────────────────────────────────────────────── */
#lumi-header {
    text-align: center;
    padding: 28px 0 12px;
    background: linear-gradient(160deg, #e8f4fd 0%, #daeeff 50%, #f0f6ff 100%);
    border-radius: 20px;
    margin-bottom: 4px;
    box-shadow: 0 2px 12px rgba(100, 160, 220, 0.12);
}
#lumi-header h1 {
    font-size: 2.0rem !important;
    font-weight: 700;
    color: #1a5f8a;
    letter-spacing: -0.3px;
    margin: 0 0 4px;
}
#lumi-header p, #lumi-header em {
    font-size: 0.95rem;
    color: #5a8aaf;
}

/* ── Avatar ─────────────────────────────────────────────────── */
#avatar-img img {
    border-radius: 50% !important;
    max-width: 180px !important;
    max-height: 180px !important;
    object-fit: cover !important;
    margin: auto;
    display: block;
    border: 4px solid #b8ddf5;
    box-shadow: 0 4px 18px rgba(100, 160, 220, 0.25);
}
#avatar-img {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#avatar-video {
    border-radius: 16px;
    max-width: 260px;
    margin: auto;
    display: block;
    box-shadow: 0 4px 18px rgba(100, 160, 220, 0.22);
}
.profile-desc {
    font-size: 1.0rem !important;
    color: #3a7dad !important;
    text-align: center;
    margin-top: 8px;
    font-style: italic;
}

/* ── Tabs ────────────────────────────────────────────────────── */
.tab-nav button {
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    color: #4a7fa0 !important;
    border-radius: 30px 30px 0 0 !important;
    padding: 10px 20px !important;
    background: #eaf4fb !important;
    border: none !important;
    transition: all 0.2s ease;
}
.tab-nav button.selected, .tab-nav button:hover {
    background: #ffffff !important;
    color: #1a5f8a !important;
    box-shadow: 0 -2px 8px rgba(100,160,220,0.15) !important;
}

/* ── Chatbot ─────────────────────────────────────────────────── */
.message-wrap {
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
}
.message.user .bubble-wrap {
    background: #d6ecfa !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #1a3a52 !important;
}
.message.bot .bubble-wrap {
    background: #ffffff !important;
    border-radius: 18px 18px 18px 4px !important;
    color: #1a3a52 !important;
    border: 1.5px solid #cce5f7 !important;
    box-shadow: 0 2px 8px rgba(100,160,220,0.09) !important;
}

/* ── Input textbox ───────────────────────────────────────────── */
textarea, input[type=text] {
    font-size: 1.1rem !important;
    border-radius: 14px !important;
    border: 1.5px solid #b8d8f0 !important;
    background: #ffffff !important;
    color: #1a3a52 !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s;
}
textarea:focus, input[type=text]:focus {
    border-color: #5aabdf !important;
    box-shadow: 0 0 0 3px rgba(90,171,223,0.15) !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #5aabdf 0%, #2a85c7 100%) !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border-radius: 30px !important;
    border: none !important;
    padding: 12px 28px !important;
    box-shadow: 0 3px 12px rgba(42,133,199,0.28) !important;
    transition: transform 0.15s, box-shadow 0.15s;
    cursor: pointer;
}
.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(42,133,199,0.38) !important;
}
.gr-button-secondary, button.secondary {
    background: #e8f4fd !important;
    color: #1a5f8a !important;
    font-size: 1.0rem !important;
    border-radius: 20px !important;
    border: 1.5px solid #b8d8f0 !important;
}

/* ── Audio/mic recorder ─────────────────────────────────────── */
.audio-component .record-button {
    background: linear-gradient(135deg, #5aabdf 0%, #2a85c7 100%) !important;
    border-radius: 50% !important;
    width: 56px !important;
    height: 56px !important;
    font-size: 1.4rem !important;
}

/* ── Status badge ────────────────────────────────────────────── */
.status-badge {
    font-size: 1.0rem !important;
    color: #4a7fa0 !important;
    text-align: center;
    padding: 6px;
    font-style: italic;
}

/* ── Radio group (profile selector) ─────────────────────────── */
.gr-radio label {
    font-size: 1.05rem !important;
    padding: 10px 18px !important;
    border-radius: 20px !important;
    border: 1.5px solid #c5dfef !important;
    background: #f0f8ff !important;
    margin: 4px !important;
    cursor: pointer;
    transition: all 0.15s;
}
.gr-radio label:hover {
    background: #d6edfb !important;
    border-color: #7ac0e8 !important;
}
.gr-radio input[type=radio]:checked + span, .gr-radio label.selected {
    background: linear-gradient(135deg, #5aabdf 0%, #2a85c7 100%) !important;
    color: #ffffff !important;
    border-color: #2a85c7 !important;
}

/* ── Panel / block styling ───────────────────────────────────── */
.gr-panel, .gradio-box, .block {
    border-radius: 18px !important;
    border: 1px solid #daeeff !important;
    background: #ffffff !important;
}

/* ── Examples ────────────────────────────────────────────────── */
.examples-table td {
    font-size: 1.0rem !important;
    color: #2a6a98 !important;
    background: #f0f8ff !important;
    border-radius: 12px !important;
    padding: 8px 14px !important;
    border: 1px solid #c5dfef !important;
    cursor: pointer;
    transition: background 0.15s;
}
.examples-table td:hover {
    background: #d6edfb !important;
}

/* ── Markdown text ───────────────────────────────────────────── */
.gr-markdown, .md {
    font-size: 1.05rem !important;
    color: #1a3a52 !important;
    line-height: 1.7 !important;
}
.gr-markdown h3 {
    color: #1a5f8a !important;
    font-size: 1.2rem !important;
    border-bottom: 1.5px solid #c5dfef;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

/* ── Footer ─────────────────────────────────────────────────── */
footer { display: none !important; }
"""

_lumi_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Lato"), "Georgia", "serif"],
    font_mono=["Courier New", "monospace"],
).set(
    body_background_fill="#f0f6ff",
    background_fill_primary="#ffffff",
    background_fill_secondary="#eaf4fb",
    border_color_primary="#c5dfef",
    color_accent_soft="#d6edfb",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
    button_primary_background_fill_hover="*primary_600",
    block_radius="18px",
    block_shadow="0 2px 12px rgba(100,160,220,0.10)",
    input_radius="14px",
    input_border_color="#b8d8f0",
    checkbox_border_radius="8px",
    prose_text_size="*text_lg",
    prose_header_text_weight="700",
)

with gr.Blocks(title="Lumi — Voice Companion", theme=_lumi_theme, css=CSS) as demo:

    # ── Persistent state ────────────────────────────────────────────────────
    profile_state = gr.State(DEFAULT_PROFILE_ID)

    # ── Header ──────────────────────────────────────────────────────────────
    with gr.Row(elem_id="lumi-header"):
        gr.Markdown(
            f"# Lumi — Your Voice Companion\n"
            f"*Good day, {PATIENT_NAME}! I'm here whenever you'd like to chat.*"
        )

    # ── Avatar area: portrait (idle) + talking video (speaking) ─────────────
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            avatar_img = gr.Image(
                value=_load_portrait(DEFAULT_PROFILE_ID) or AVATAR_IMAGES["smile"],
                label="",
                show_label=False,
                elem_id="avatar-img",
                interactive=False,
                width=180,
                height=180,
                visible=True,
            )
            avatar_video = gr.Video(
                label="",
                show_label=False,
                elem_id="avatar-video",
                autoplay=True,
                visible=False,
                width=260,
                height=260,
            )
            profile_desc_md = gr.Markdown(
                f"**{get_profile(DEFAULT_PROFILE_ID)['display_name']}** — "
                f"{get_profile(DEFAULT_PROFILE_ID)['description']}",
                elem_classes=["profile-desc"],
            )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # Tab 1 — Chat ────────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            chatbot1 = gr.Chatbot(height=380, label="",
                                   show_copy_button=False, show_share_button=False,
                                   type="messages")
            with gr.Row():
                msg1  = gr.Textbox(
                    placeholder="Type something to Lumi…", scale=7, container=False, lines=1
                )
                send1 = gr.Button("Send ➤", scale=1, variant="primary")

            gr.Examples(
                examples=[
                    "I can't remember where I put my glasses.",
                    "What did we talk about last time?",
                    "I'm feeling a bit lonely today.",
                    "Tell me something cheerful.",
                ],
                inputs=msg1,
                label="💡 Try saying...",
            )

            send1.click(
                text_chat, [msg1, chatbot1, profile_state],
                [chatbot1, avatar_img],
            ).then(lambda: "", None, msg1)
            msg1.submit(
                text_chat, [msg1, chatbot1, profile_state],
                [chatbot1, avatar_img],
            ).then(lambda: "", None, msg1)

        # Tab 2 — Voice Chat ────────────────────────────────────
        with gr.Tab("🎤 Voice"):
            gr.Markdown(
                "🎤 **Record your message, then click Send.** "
                "Lumi will listen, think, and speak back."
            )
            chatbot2 = gr.Chatbot(height=260, label="",
                                   show_copy_button=False, show_share_button=False,
                                   type="messages")

            with gr.Row():
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎙 Record your message",
                )
                voice_send = gr.Button("🎤 Send", variant="primary", scale=0)

            audio_live_out = gr.Audio(label="Lumi's voice", autoplay=True, visible=True)
            video_live_out = gr.Video(
                label="", show_label=False, autoplay=True,
                elem_id="avatar-video", visible=False,
            )
            status_live = gr.Markdown("🎤 Record then click Send", elem_classes=["status-badge"])

            voice_send.click(
                voice_submit,
                inputs=[mic_input, chatbot2, profile_state],
                outputs=[chatbot2, audio_live_out, video_live_out, avatar_img, status_live],
            )

            # When a video comes back: hide static avatar, show video
            def _show_video(v):
                if v:
                    return gr.update(visible=False), gr.update(visible=True, value=v)
                return gr.update(visible=True), gr.update(visible=False)

            video_live_out.change(
                _show_video, [video_live_out], [avatar_img, video_live_out]
            )

        # Tab 3 — Profiles ───────────────────────────────────────────────────
        with gr.Tab("🎭 Choose Companion"):
            gr.Markdown("### Select your companion's profile")
            gr.Markdown(
                "Each profile has a unique appearance, voice, and speaking style. "
                "Your choice applies to both text and voice chat."
            )
            profile_radio = gr.Radio(
                choices=profile_labels(),
                value=get_profile(DEFAULT_PROFILE_ID)["label"],
                label="Profile",
            )
            profile_info = gr.Markdown(
                f"**{get_profile(DEFAULT_PROFILE_ID)['display_name']}** — "
                f"{get_profile(DEFAULT_PROFILE_ID)['description']}",
            )
            gr.Markdown(
                "> **Tip:** The selected profile's voice and portrait apply to both "
                "text chat and voice chat. Switch anytime — it takes effect on the next message.",
                visible=True,
            )

            profile_radio.change(
                change_profile,
                [profile_radio],
                [profile_state, avatar_img, profile_desc_md],
            )

        # Tab 4 — Family Dashboard ───────────────────────────────────────────
        with gr.Tab("👨‍👩‍👧 Family Dashboard"):
            gr.Markdown("### Session Summary")
            summary_btn = gr.Button("Generate Summary", variant="primary")
            summary_out = gr.Textbox(label="Session Summary Card", lines=12, interactive=False)
            summary_btn.click(end_session, [chatbot1], [summary_out])

            gr.Markdown("---")
            gr.Markdown("### Lumi's Memory")
            refresh_btn = gr.Button("Refresh Memory", size="sm")
            with gr.Row():
                facts_box    = gr.Textbox(label="Known Facts",                lines=6, interactive=False)
                sessions_box = gr.Textbox(label="Previous Session Summaries", lines=6, interactive=False)
            refresh_btn.click(load_memory_display, [], [facts_box, sessions_box])

        # Tab 5 — About ──────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## About Lumi

Lumi is a fine-tuned AI voice companion for elderly patients with dementia and Alzheimer's.

**What makes Lumi different:**
- **Domain fine-tuned** — Qwen3-4B on 8,540 dementia-care conversations (SFT + GRPO)
- **10 diverse profiles** — choose a companion that feels right for the patient
- **Talking head avatar** — SadTalker on AMD MI300X: lip sync, expressions, head motion
- **Persistent memory** — ChromaDB remembers personal details across sessions
- **Scam protection** — detects and deflects elder fraud attempts
- **Continuous voice** — VAD-driven live conversation, no re-recording between turns
- **AMD MI300X** — fine-tuned and served on AMD hardware via ROCm + vLLM

**Patient:** {PATIENT_NAME} | **Model:** {MODEL_NAME}

**EQ-Bench score: 91.22/100** · **Latency: 21ms median TTFA** on AMD MI300X

Built for the AMD Developer Hackathon 2026 · Fine-Tuning Track
""")

    demo.load(load_memory_display, [], [facts_box, sessions_box])

    # ── React frontend API endpoint (hidden, called by @gradio/client) ───────
    _msg   = gr.Textbox(visible=False)
    _hist  = gr.JSON(visible=False)
    _prof  = gr.Textbox(visible=False, value=DEFAULT_PROFILE_ID)
    _resp  = gr.Textbox(visible=False)
    _audio = gr.Textbox(visible=False)
    _tag   = gr.Textbox(visible=False)
    _btn   = gr.Button(visible=False)
    _btn.click(
        fn=lumi_api,
        inputs=[_msg, _hist, _prof],
        outputs=[_resp, _audio, _tag],
        api_name="lumi_api",
    )

# ---------------------------------------------------------------------------
# FastAPI wrapper — serves React at / and Gradio API at /gradio
# ---------------------------------------------------------------------------

DIST_DIR = Path(__file__).parent / "dist"

_fastapi = FastAPI()

from pipeline.memory import get_all_summaries

@_fastapi.get("/api/summaries")
async def fetch_summaries():
    # In a real app, you'd use the user ID from auth.
    # For the hackathon, we use the global PATIENT_ID.
    return get_all_summaries(PATIENT_ID)

@_fastapi.get("/api/whoami")
async def whoami(request: Request):
    # Check multiple possible HF headers (request-specific)
    user = (
        request.headers.get("X-HF-User") or 
        request.headers.get("X-HF-User-Name") or 
        request.headers.get("X-Forwarded-User") or 
        "Guest"
    )
    return {"username": user}

# Mount Gradio FIRST so its /gradio/* routes have priority over the catch-all.
os.environ.setdefault("GRADIO_ALLOWED_PATHS", "/tmp")
app = gr.mount_gradio_app(_fastapi, demo, path="/gradio")

if DIST_DIR.exists():
    for _subdir in ("assets", "avatars"):
        _path = DIST_DIR / _subdir
        if _path.exists():
            _fastapi.mount(f"/{_subdir}", StaticFiles(directory=str(_path)), name=_subdir)

    @_fastapi.get("/")
    async def _root():
        return FileResponse(str(DIST_DIR / "index.html"))

    @_fastapi.get("/app")
    async def _app_route():
        return FileResponse(str(DIST_DIR / "index.html"))

    @_fastapi.get("/{full_path:path}")
    async def _spa_fallback(full_path: str):
        candidate = DIST_DIR / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(DIST_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

# Force Update: 2026-05-10T00:23:00

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
import time
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

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


def _build_messages(history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": _get_system_prompt()}] + history


def _tts_and_video(text: str, profile_id: str) -> tuple[str | None, str | None]:
    """Run TTS and optionally SadTalker. Returns (audio_path, video_path)."""
    profile    = get_profile(profile_id)
    audio_path = synthesize(text, voice=profile["voice"],
                            rate=profile["rate"], pitch=profile["pitch"])
    video_path = None
    if audio_path and talking_head_available():
        video_path = generate_talking_video(audio_path, profile_id)
    return audio_path, video_path


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
    is_scam, deflection = check_and_deflect(message)
    if is_scam:
        new_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": deflection},
        ]
        yield new_history, AVATAR_IMAGES["gentle"]
        return

    messages = _build_messages(history) + [{"role": "user", "content": message}]
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
    if facts:
        save_session(PATIENT_ID, facts, "unknown", "unknown",
                     f"Text session — {len(new_history)//2} turns")


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

        facts = extract_facts_from_response(raw)
        if facts:
            save_session(PATIENT_ID, facts, "unknown", "unknown", "Voice session")
    except Exception as e:
        print(f"[LLM/TTS] {e}")
        response_text = "I'm sorry, I had a little trouble. Could you try again?"
        audio_path, video_path = _tts_and_video(response_text, profile_id)

    new_history = history + [
        {"role": "user",      "content": f"🎤 {user_text}"},
        {"role": "assistant", "content": response_text},
    ]
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
#avatar-img   { border-radius: 50%; max-width: 200px; margin: auto; display: block; }
#avatar-video { border-radius: 16px; max-width: 280px; margin: auto; display: block; }
#lumi-header  { text-align: center; }
.status-badge { font-size: 0.85rem; color: #555; text-align: center; }
.profile-desc { font-size: 0.85rem; color: #666; text-align: center; margin-top: 4px; }
"""

with gr.Blocks(title="Lumi — Voice Companion", theme=gr.themes.Soft(), css=CSS) as demo:

    # ── Persistent state ────────────────────────────────────────────────────
    profile_state = gr.State(DEFAULT_PROFILE_ID)

    # ── Header ──────────────────────────────────────────────────────────────
    with gr.Row(elem_id="lumi-header"):
        gr.Markdown(
            "# Lumi — AI Voice Companion\n"
            "*Powered by AMD MI300X · Fine-tuned Qwen3-4B (SFT + GRPO)*"
        )

    # ── Avatar area: static portrait (idle) + talking video (speaking) ──────
    with gr.Row():
        with gr.Column(scale=1):
            avatar_img = gr.Image(
                value=_load_portrait(DEFAULT_PROFILE_ID) or AVATAR_IMAGES["smile"],
                label="",
                show_label=False,
                elem_id="avatar-img",
                interactive=False,
                show_download_button=False,
                show_fullscreen_button=False,
                width=200,
                height=200,
                visible=True,
            )
            avatar_video = gr.Video(
                label="",
                show_label=False,
                elem_id="avatar-video",
                autoplay=True,
                visible=False,
                width=280,
                height=280,
            )
            profile_desc_md = gr.Markdown(
                f"**{get_profile(DEFAULT_PROFILE_ID)['display_name']}** — "
                f"{get_profile(DEFAULT_PROFILE_ID)['description']}",
                elem_classes=["profile-desc"],
            )

    # ── Tabs ────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # Tab 1 — Text Chat ──────────────────────────────────────────────────
        with gr.Tab("💬 Text Chat"):
            chatbot1 = gr.Chatbot(height=400, label="Conversation", type="messages",
                                   show_copy_button=False, show_share_button=False)
            with gr.Row():
                msg1  = gr.Textbox(
                    placeholder="Type a message to Lumi…", scale=7, container=False
                )
                send1 = gr.Button("Send", scale=1, variant="primary")

            gr.Examples(
                examples=[
                    "I can't remember where I put my glasses.",
                    "What did we talk about last time?",
                    "I'm feeling a bit lonely today.",
                    "Tell me something cheerful.",
                ],
                inputs=msg1,
            )

            send1.click(
                text_chat, [msg1, chatbot1, profile_state],
                [chatbot1, avatar_img],
            ).then(lambda: "", None, msg1)
            msg1.submit(
                text_chat, [msg1, chatbot1, profile_state],
                [chatbot1, avatar_img],
            ).then(lambda: "", None, msg1)

        # Tab 2 — Voice Chat (record → Send) ──────────────────────────────────
        with gr.Tab("🎤 Voice Chat"):
            gr.Markdown(
                "**Record your message, then click Send.** "
                "Lumi will transcribe, think, and speak back."
            )
            chatbot2 = gr.Chatbot(height=280, label="Conversation", type="messages",
                                   show_copy_button=False, show_share_button=False)

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

"""
Lumi — AI Voice Companion for Dementia & Alzheimer's patients.
HuggingFace Spaces Gradio app.

Environment variables (set as HF Space Secrets):
  VLLM_BASE_URL  — vLLM OpenAI-compatible endpoint, e.g. http://IP:8000/v1
  MODEL_NAME     — model ID served by vLLM, e.g. YUGOROU/lumi-qwen3-4b
  STT_URL        — Whisper-compatible STT endpoint (optional; leave blank for text-only)
  TTS_URL        — TTS endpoint compatible with /v1/audio/speech (optional)
  PATIENT_ID     — persistent ID for ChromaDB memory (default: demo_user_001)
  PATIENT_NAME   — patient's first name shown in the UI
"""

import os
import time
import tempfile
import numpy as np
import soundfile as sf
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from openai import OpenAI

from pipeline.memory import build_system_prompt, get_context, save_session
from pipeline.parser import extract_facts_from_response, parse_structured_output
from pipeline.scam_filter import check_and_deflect
from pipeline.stt import transcribe
from pipeline.tts import synthesize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Debdeep30/lumi-qwen3-4b-grpo")
PATIENT_ID    = os.getenv("PATIENT_ID", "demo_user_001")
PATIENT_NAME  = os.getenv("PATIENT_NAME", "Margaret")
STT_URL       = os.getenv("STT_URL", "")
TTS_URL       = os.getenv("TTS_URL", "")

llm = OpenAI(base_url=VLLM_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "not-required"))

AVATAR_DIR = os.path.join(os.path.dirname(__file__), "avatar")
AVATAR_IMAGES = {
    tag: Image.open(os.path.join(AVATAR_DIR, f"avatar_{tag}.png"))
    for tag in ("smile", "nod", "concerned", "gentle", "laugh")
}

# VAD tuning
SILENCE_THRESHOLD = 0.015   # normalised RMS below which a chunk is "silent"
SILENCE_CHUNKS    = 6       # consecutive silent chunks → end of utterance (~1.5 s)
MIN_SPEECH_CHUNKS = 3       # ignore clips shorter than this (~0.75 s)

# ---------------------------------------------------------------------------
# Core conversation logic
# ---------------------------------------------------------------------------

def _get_system_prompt() -> str:
    return build_system_prompt(PATIENT_ID, PATIENT_NAME)


def _call_llm(messages: list[dict]) -> str:
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        extra_body={"repetition_penalty": 1.2}
    )
    return resp.choices[0].message.content or ""


def _call_llm_stream(messages: list[dict]):
    for chunk in llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        stream=True,
        extra_body={"repetition_penalty": 1.2}
    ):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _build_messages(history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": _get_system_prompt()}] + history


def _process_response(raw: str) -> dict:
    parsed = parse_structured_output(raw)
    audio_path = None
    if TTS_URL:
        audio_path = synthesize(parsed["full_response"])
    return {**parsed, "audio_path": audio_path}


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
# Gradio handlers
# ---------------------------------------------------------------------------

def text_chat(message: str, history: list[dict]):
    """Text in → streamed text out."""
    is_scam, deflection = check_and_deflect(message)
    if is_scam:
        new_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": deflection},
        ]
        yield new_history, AVATAR_IMAGES["gentle"]
        return

    messages = _build_messages(history) + [{"role": "user", "content": message}]
    partial = ""
    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]
    for token in _call_llm_stream(messages):
        partial += token
        parsed = parse_structured_output(partial)
        new_history[-1]["content"] = parsed["full_response"]
        yield new_history, AVATAR_IMAGES.get(parsed["avatar_tag"], AVATAR_IMAGES["smile"])

    facts = extract_facts_from_response(partial)
    if facts:
        save_session(PATIENT_ID, facts, "unknown", "unknown",
                     f"Text session — {len(new_history)//2} turns")


# ---------------------------------------------------------------------------
# Live voice — streaming VAD
# ---------------------------------------------------------------------------

_LIVE_STATE_DEFAULT = {
    "buffer":        [],    # accumulated audio chunks (numpy arrays)
    "sr":            16000, # sample rate (updated on first chunk)
    "silent_chunks": 0,
    "is_speaking":   False,
    "history":       [],
}


def _rms(audio: np.ndarray) -> float:
    a = audio.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(a ** 2)))


def _save_wav(chunks: list[np.ndarray], sr: int) -> str:
    audio = np.concatenate(chunks)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    return tmp.name


def live_stream(chunk, state: dict):
    """
    Called on every ~250 ms audio chunk from the microphone.
    Returns: (new_state, tts_audio_path, chatbot_history, status_md, avatar_img)
    """
    no_change = (state, None, state["history"], "🎙 Listening…", AVATAR_IMAGES["smile"])

    if chunk is None:
        return no_change

    sr, audio = chunk
    state = dict(state)          # shallow copy so Gradio detects change
    state["sr"] = int(sr)

    energy = _rms(audio)
    state["buffer"] = state["buffer"] + [audio]

    if energy > SILENCE_THRESHOLD:
        state["is_speaking"]   = True
        state["silent_chunks"] = 0
        return state, None, state["history"], "🗣 Listening — speak freely…", AVATAR_IMAGES["nod"]

    # Silent chunk
    state["silent_chunks"] = state["silent_chunks"] + 1

    if not state["is_speaking"] or state["silent_chunks"] < SILENCE_CHUNKS:
        return state, None, state["history"], "🎙 Listening…", AVATAR_IMAGES["smile"]

    # ── End of utterance detected ──────────────────────────────────────────
    speech_chunks = state["buffer"][: -state["silent_chunks"]]  # trim trailing silence
    state["buffer"]        = []
    state["silent_chunks"] = 0
    state["is_speaking"]   = False

    if len(speech_chunks) < MIN_SPEECH_CHUNKS:
        return state, None, state["history"], "🎙 Listening…", AVATAR_IMAGES["smile"]

    # STT
    audio_path = _save_wav(speech_chunks, state["sr"])
    user_text  = ""
    try:
        user_text = transcribe(audio_path) if STT_URL else "[STT not configured]"
    except Exception as e:
        print(f"[STT] {e}")
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    if not user_text:
        return state, None, state["history"], "🎙 Listening…", AVATAR_IMAGES["smile"]

    # Scam check
    is_scam, deflection = check_and_deflect(user_text)
    if is_scam:
        tts_path = synthesize(deflection) if TTS_URL else None
        new_history = state["history"] + [
            {"role": "user",      "content": f"🎤 {user_text}"},
            {"role": "assistant", "content": deflection},
        ]
        state["history"] = new_history
        return state, tts_path, new_history, "🎙 Listening…", AVATAR_IMAGES["gentle"]

    # LLM + TTS
    messages = _build_messages(state["history"]) + [{"role": "user", "content": user_text}]
    response_text = ""
    tts_path      = None
    avatar        = AVATAR_IMAGES["smile"]
    try:
        raw    = _call_llm(messages)
        parsed = parse_structured_output(raw)
        response_text = parsed["full_response"]
        avatar        = AVATAR_IMAGES.get(parsed["avatar_tag"], AVATAR_IMAGES["smile"])
        tts_path      = synthesize(response_text) if TTS_URL else None

        facts = extract_facts_from_response(raw)
        if facts:
            save_session(PATIENT_ID, facts, "unknown", "unknown", "Live voice session")
    except Exception as e:
        print(f"[LLM/TTS] {e}")
        response_text = "I'm sorry, I had a little trouble just then. Could you say that again?"
        tts_path = synthesize(response_text) if TTS_URL else None

    new_history = state["history"] + [
        {"role": "user",      "content": f"🎤 {user_text}"},
        {"role": "assistant", "content": response_text},
    ]
    state["history"] = new_history
    return state, tts_path, new_history, "🎙 Listening…", avatar


def end_session(history: list[dict]):
    if not history:
        return "No conversation to summarise."
    summary = _end_of_session_summary(history)
    date = time.strftime("%B %d, %Y")
    return (
        f"SESSION SUMMARY — {date}\n"
        f"Patient: {PATIENT_NAME}\n"
        f"Duration: {len(history)//2} turns\n\n"
        f"{summary}"
    )


def load_memory_display():
    ctx = get_context(PATIENT_ID)
    facts_txt    = "\n".join(f"• {f}" for f in ctx["facts"]) or "(none yet)"
    sessions_txt = "\n\n---\n\n".join(ctx["summaries"]) or "(no previous sessions)"
    return facts_txt, sessions_txt


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#avatar-img { border-radius: 50%; max-width: 200px; margin: auto; display: block; }
#lumi-header { text-align: center; }
.status-badge { font-size: 0.85rem; color: #555; text-align: center; }
"""

with gr.Blocks(title="Lumi — Voice Companion", theme=gr.themes.Soft(), css=CSS) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    with gr.Row(elem_id="lumi-header"):
        gr.Markdown("# Lumi — AI Voice Companion\n*Powered by AMD MI300X · Fine-tuned Qwen3-4B (SFT + GRPO)*")

    # ── Avatar ──────────────────────────────────────────────────────────────
    avatar_img = gr.Image(
        value=AVATAR_IMAGES["smile"],
        label="Lumi",
        show_label=False,
        elem_id="avatar-img",
        interactive=False,
        width=200,
        height=200,
    )

    # ── Tabs ────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # Tab 1 — Text Chat
        with gr.Tab("💬 Text Chat"):
            chatbot1 = gr.Chatbot(
                height=420,
                label="Conversation",
                type="messages",
                avatar_images=(None, AVATAR_IMAGES["smile"]),
            )
            with gr.Row():
                msg1  = gr.Textbox(
                    placeholder="Type a message to Lumi…",
                    scale=7,
                    container=False,
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

            send1.click(text_chat, [msg1, chatbot1], [chatbot1, avatar_img]).then(
                lambda: "", None, msg1
            )
            msg1.submit(text_chat, [msg1, chatbot1], [chatbot1, avatar_img]).then(
                lambda: "", None, msg1
            )

        # Tab 2 — Live Voice Chat
        with gr.Tab("🎤 Live Voice Chat"):
            gr.Markdown(
                "**Click the microphone button once to start.** "
                "Speak naturally — Lumi listens automatically and responds when you pause. "
                "Click the button again to stop the session."
            )

            chatbot2 = gr.Chatbot(height=300, label="Conversation", type="messages")

            mic_live = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="🎙 Microphone",
            )
            audio_live_out = gr.Audio(label="Lumi's voice", autoplay=True)
            status_live = gr.Markdown("🎙 Listening…", elem_classes=["status-badge"])

            live_state = gr.State(dict(_LIVE_STATE_DEFAULT))

            mic_live.stream(
                live_stream,
                inputs=[mic_live, live_state],
                outputs=[live_state, audio_live_out, chatbot2, status_live, avatar_img],
            )

        # Tab 3 — Family Dashboard
        with gr.Tab("👨‍👩‍👧 Family Dashboard"):
            gr.Markdown("### Session Summary")
            gr.Markdown(
                "Click **Generate Summary** after a session ends to create a "
                "family-readable report."
            )
            summary_btn = gr.Button("Generate Summary", variant="primary")
            summary_out = gr.Textbox(
                label="Session Summary Card",
                lines=12,
                interactive=False,
            )
            summary_btn.click(end_session, [chatbot1], [summary_out])

            gr.Markdown("---")
            gr.Markdown("### Lumi's Memory")
            refresh_btn = gr.Button("Refresh Memory", size="sm")
            with gr.Row():
                facts_box    = gr.Textbox(label="Known Facts",                lines=6, interactive=False)
                sessions_box = gr.Textbox(label="Previous Session Summaries", lines=6, interactive=False)
            refresh_btn.click(load_memory_display, [], [facts_box, sessions_box])

        # Tab 4 — About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## About Lumi

Lumi is a fine-tuned AI voice companion designed for elderly patients with dementia and Alzheimer's disease.

**What makes Lumi different:**
- **Domain fine-tuned** — Qwen3-4B fine-tuned on 8,540 dementia-care conversations via SFT + GRPO (two-phase RL)
- **Persistent memory** — remembers personal details across sessions using ChromaDB
- **Scam protection** — detects and deflects elder fraud attempts without alarming the patient
- **Voice-native** — Whisper STT → Lumi → TTS, continuous live conversation
- **AMD MI300X** — fine-tuned and served on AMD hardware via ROCm + vLLM

**Patient:** {PATIENT_NAME} | **Model:** {MODEL_NAME}

**EQ-Bench score: 91.22/100** (base model: 0.00) · **Latency: 21ms median TTFA** on AMD MI300X

Built for the AMD Developer Hackathon 2026 · Fine-Tuning Track
""")

    demo.load(load_memory_display, [], [facts_box, sessions_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

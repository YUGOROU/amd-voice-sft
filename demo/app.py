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
import sys
import time
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from openai import OpenAI

# make pipeline importable when running from demo/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.memory import build_system_prompt, get_context, save_session
from pipeline.parser import extract_facts_from_response, parse_structured_output
from pipeline.scam_filter import check_and_deflect
from pipeline.stt import transcribe
from pipeline.tts import synthesize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "YUGOROU/lumi-gemma-4-31b")
PATIENT_ID    = os.getenv("PATIENT_ID", "demo_user_001")
PATIENT_NAME  = os.getenv("PATIENT_NAME", "Margaret")
STT_URL       = os.getenv("STT_URL", "")
TTS_URL       = os.getenv("TTS_URL", "")

llm = OpenAI(base_url=VLLM_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "not-required"))

# Avatar image paths (relative to this file)
AVATAR_DIR = os.path.join(os.path.dirname(__file__), "avatar")
AVATAR_IMAGES = {
    tag: os.path.join(AVATAR_DIR, f"avatar_{tag}.png")
    for tag in ("smile", "nod", "concerned", "gentle", "laugh")
}

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
    """Parse structured output and queue TTS for the opening line."""
    parsed = parse_structured_output(raw)
    audio_path = None
    if TTS_URL:
        audio_path = synthesize(parsed["opening_line"])
    return {**parsed, "audio_path": audio_path}


def _end_of_session_summary(history: list[dict]) -> str:
    """Generate a family-readable session summary via LLM."""
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
    """Mode 1 — text in, streamed text out (no TTS)."""
    is_scam, deflection = check_and_deflect(message)
    if is_scam:
        new_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": deflection},
        ]
        yield new_history
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

    # persist facts
    facts = extract_facts_from_response(partial)
    if facts:
        save_session(PATIENT_ID, facts, "unknown", "unknown",
                     f"Text session — {len(new_history)//2} turns")


def voice_chat(audio_path: str | None, history: list[dict]):
    """Mode 2 — mic in → STT → LLM → TTS → avatar."""
    if not audio_path:
        return history, None, ""

    t0 = time.time()
    user_text = transcribe(audio_path) if STT_URL else "[STT not configured]"
    if not user_text:
        return history, None, ""

    is_scam, deflection = check_and_deflect(user_text)
    if is_scam:
        audio_out = synthesize(deflection) if TTS_URL else None
        new_history = history + [
            {"role": "user",      "content": f"🎤 {user_text}"},
            {"role": "assistant", "content": deflection},
        ]
        return new_history, audio_out, f"⏱ {time.time()-t0:.1f}s", AVATAR_IMAGES["gentle"]

    messages = _build_messages(history) + [{"role": "user", "content": user_text}]
    raw = _call_llm(messages)
    parsed = _process_response(raw)

    latency = time.time() - t0
    new_history = history + [
        {"role": "user",      "content": f"🎤 {user_text}"},
        {"role": "assistant", "content": parsed["full_response"]},
    ]

    facts = extract_facts_from_response(raw)
    if facts:
        save_session(PATIENT_ID, facts, "unknown", "unknown",
                     f"Voice session — {len(new_history)//2} turns")

    return new_history, parsed["audio_path"], f"⏱ {latency:.1f}s", AVATAR_IMAGES.get(parsed["avatar_tag"], AVATAR_IMAGES["smile"])


def end_session(history: list[dict]):
    """Generate family summary card at end of session."""
    if not history:
        return "No conversation to summarise."
    summary = _end_of_session_summary(history)
    date = time.strftime("%B %d, %Y")
    card = (
        f"SESSION SUMMARY — {date}\n"
        f"Patient: {PATIENT_NAME}\n"
        f"Duration: {len(history)//2} turns\n\n"
        f"{summary}"
    )
    return card


def load_memory_display():
    ctx = get_context(PATIENT_ID)
    facts_txt = "\n".join(f"• {f}" for f in ctx["facts"]) or "(none yet)"
    sessions_txt = "\n\n---\n\n".join(ctx["summaries"]) or "(no previous sessions)"
    return facts_txt, sessions_txt


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#avatar-img { border-radius: 50%; max-width: 200px; margin: auto; display: block; }
#lumi-header { text-align: center; }
.latency-badge { font-size: 0.75rem; color: #888; }
"""

with gr.Blocks(title="Lumi — Voice Companion", theme=gr.themes.Soft(), css=CSS) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    with gr.Row(elem_id="lumi-header"):
        gr.Markdown("# Lumi — AI Voice Companion\n*Powered by AMD MI300X · Fine-tuned Gemma-4-31B*")

    # ── Avatar ──────────────────────────────────────────────────────────────
    avatar_img = gr.Image(
        value=AVATAR_IMAGES.get("smile"),
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
                avatar_images=(None, AVATAR_IMAGES.get("smile")),
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

        # Tab 2 — Voice Chat
        with gr.Tab("🎤 Voice Chat"):
            chatbot2 = gr.Chatbot(
                height=360,
                label="Conversation",
            )
            audio_in  = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Speak to Lumi",
            )
            audio_out = gr.Audio(
                label="Lumi's voice",
                autoplay=True,
            )
            latency_badge = gr.Markdown("", elem_classes=["latency-badge"])

            audio_in.stop_recording(
                voice_chat,
                [audio_in, chatbot2],
                [chatbot2, audio_out, latency_badge, avatar_img],
            )

            if not STT_URL:
                gr.Markdown(
                    "> **Note:** `STT_URL` not configured — voice transcription disabled. "
                    "Set the Space secret to enable full voice mode.",
                    visible=True,
                )
            if not TTS_URL:
                gr.Markdown(
                    "> **Note:** `TTS_URL` not configured — voice output disabled.",
                    visible=True,
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
                facts_box    = gr.Textbox(
                    label="Known Facts",
                    lines=6,
                    interactive=False,
                )
                sessions_box = gr.Textbox(
                    label="Previous Session Summaries",
                    lines=6,
                    interactive=False,
                )
            refresh_btn.click(load_memory_display, [], [facts_box, sessions_box])

        # Tab 4 — About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## About Lumi

Lumi is a fine-tuned AI voice companion designed for elderly patients with dementia and Alzheimer's disease.

**What makes Lumi different:**
- **Domain fine-tuned** — Gemma-4-31B fine-tuned on 8,500+ dementia-care conversations via EQ-Matrix pipeline
- **Persistent memory** — remembers personal details across sessions using ChromaDB
- **Scam protection** — detects and deflects elder fraud attempts without alarming the patient
- **Voice-native** — Whisper STT → Lumi → TTS, <1.5s time-to-first-audio
- **AMD MI300X** — fine-tuned and served on AMD hardware via ROCm + vLLM

**Patient:** {PATIENT_NAME} | **Model:** {MODEL_NAME}

Built for the AMD Developer Hackathon 2026 · Fine-Tuning Track
""")

    # load memory on startup
    demo.load(load_memory_display, [], [facts_box, sessions_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

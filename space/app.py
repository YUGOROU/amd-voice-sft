import os
import tempfile

import gradio as gr
import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — set these as HF Space Secrets (Settings → Variables and secrets)
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
STT_URL       = os.getenv("STT_URL", "")   # Granite Speech endpoint  e.g. http://IP:8001
TTS_URL       = os.getenv("TTS_URL", "")   # VoxCPM2 endpoint          e.g. http://IP:8002

SYSTEM_PROMPT = (
    "You are Lumi, a warm and patient AI companion for elderly users with memory difficulties. "
    "Always respond with warmth, patience, and clarity. Keep your responses short and simple. "
    "Never use clinical or formal language."
)

llm = OpenAI(base_url=VLLM_BASE_URL, api_key="not-required")


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _make_messages(history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def _transcribe(audio_path: str) -> str:
    """STT via Granite Speech endpoint (OpenAI-compatible /v1/audio/transcriptions)."""
    stt = OpenAI(base_url=STT_URL.rstrip("/"), api_key="not-required")
    with open(audio_path, "rb") as f:
        return stt.audio.transcriptions.create(model="granite-speech", file=f).text


def _synthesize(text: str) -> str | None:
    """TTS via VoxCPM2 endpoint (/v1/audio/speech) → tmp WAV path."""
    r = httpx.post(
        f"{TTS_URL.rstrip('/')}/v1/audio/speech",
        json={"model": "VoxCPM2", "input": text, "response_format": "wav"},
        timeout=30.0,
    )
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(r.content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Mode 1 — Text Chat (streaming)
# ---------------------------------------------------------------------------

def text_chat(message: str, history: list[dict]):
    msgs = _make_messages(history) + [{"role": "user", "content": message}]
    partial = ""
    for chunk in llm.chat.completions.create(model=MODEL_NAME, messages=msgs, stream=True):
        delta = chunk.choices[0].delta.content
        if delta:
            partial += delta
            yield partial


# ---------------------------------------------------------------------------
# Mode 2 — Text → LLM → TTS
# ---------------------------------------------------------------------------

def text_tts_submit(message: str, history: list[dict]):
    msgs = _make_messages(history) + [{"role": "user", "content": message}]
    reply = llm.chat.completions.create(model=MODEL_NAME, messages=msgs).choices[0].message.content
    audio_path = _synthesize(reply) if TTS_URL else None
    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history, audio_path, ""


# ---------------------------------------------------------------------------
# Mode 3 — Voice Chat (STT → LLM → TTS)
# ---------------------------------------------------------------------------

def voice_chat_submit(audio_path: str | None, history: list[dict]):
    if not audio_path:
        return history, None
    user_text = _transcribe(audio_path) if STT_URL else "[STT_URL not configured]"
    msgs = _make_messages(history) + [{"role": "user", "content": user_text}]
    reply = llm.chat.completions.create(model=MODEL_NAME, messages=msgs).choices[0].message.content
    audio_out = _synthesize(reply) if TTS_URL else None
    new_history = history + [
        {"role": "user",      "content": f"🎤 {user_text}"},
        {"role": "assistant", "content": reply},
    ]
    return new_history, audio_out


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Lumi — Voice Companion", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Lumi — AI Voice Companion\nPowered by AMD MI300X · Lumi is here to listen.")

    with gr.Tab("💬 Text Chat"):
        gr.ChatInterface(
            fn=text_chat,
            type="messages",
            chatbot=gr.Chatbot(height=480, type="messages"),
            textbox=gr.Textbox(placeholder="Type a message…", scale=7),
            examples=["I forgot what I was doing.", "Tell me something nice.", "Where am I?"],
            cache_examples=False,
        )

    with gr.Tab("🔊 Text + Voice Response"):
        chatbot2 = gr.Chatbot(height=400, type="messages")
        with gr.Row():
            msg2  = gr.Textbox(placeholder="Type a message…", scale=7)
            send2 = gr.Button("Send", scale=1, variant="primary")
        audio_out2 = gr.Audio(label="Lumi's voice", autoplay=True)

        send2.click(
            fn=text_tts_submit,
            inputs=[msg2, chatbot2],
            outputs=[chatbot2, audio_out2, msg2],
        )
        msg2.submit(
            fn=text_tts_submit,
            inputs=[msg2, chatbot2],
            outputs=[chatbot2, audio_out2, msg2],
        )

    with gr.Tab("🎤 Voice Chat"):
        chatbot3 = gr.Chatbot(height=380, type="messages")
        audio_in3  = gr.Audio(sources=["microphone"], type="filepath", label="Speak to Lumi")
        audio_out3 = gr.Audio(label="Lumi's reply", autoplay=True)

        audio_in3.stop_recording(
            fn=voice_chat_submit,
            inputs=[audio_in3, chatbot3],
            outputs=[chatbot3, audio_out3],
        )

if __name__ == "__main__":
    demo.launch()

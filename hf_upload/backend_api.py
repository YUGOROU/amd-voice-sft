"""
Lumi Backend API — runs on the droplet at port 7860.

Exposes `lumi_api` as a Gradio predict endpoint so the React frontend
(served by nginx at port 80) can call it via @gradio/client.

Usage:
    cd /root/lumi/hf_upload
    pip install -r requirements.txt
    python backend_api.py
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from openai import OpenAI

from pipeline.memory      import build_system_prompt, save_session
from pipeline.parser      import extract_facts_from_response, parse_structured_output
from pipeline.profiles    import get_profile
from pipeline.scam_filter import check_and_deflect
from pipeline.tts         import synthesize

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "Debdeep30/lumi-qwen3-4b-grpo")
PATIENT_ID    = os.getenv("PATIENT_ID",   "demo_user_001")
PATIENT_NAME  = os.getenv("PATIENT_NAME", "Margaret")

llm = OpenAI(base_url=VLLM_BASE_URL, api_key="not-required")


def _call_llm(messages: list[dict]) -> str:
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        extra_body={"repetition_penalty": 1.2},
    )
    return resp.choices[0].message.content or ""


def lumi_api(message: str, history: list, profile_id: str):
    """
    Main API for the React frontend.
    Args:
        message    — user's text message
        history    — list of {role, content} dicts
        profile_id — avatar profile ID (e.g. "sophie")
    Returns:
        (response_text, audio_path, avatar_tag)
    """
    is_scam, deflection = check_and_deflect(message)
    if is_scam:
        profile = get_profile(profile_id)
        audio_path = synthesize(
            deflection,
            voice=profile["voice"],
            rate=profile["rate"],
            pitch=profile["pitch"],
        )
        return deflection, audio_path or "", "gentle"

    system_prompt = build_system_prompt(PATIENT_ID, PATIENT_NAME)
    messages = [{"role": "system", "content": system_prompt}] + list(history) + [
        {"role": "user", "content": message}
    ]

    try:
        raw           = _call_llm(messages)
        parsed        = parse_structured_output(raw)
        response_text = parsed["full_response"]
        avatar_tag    = parsed["avatar_tag"]

        profile    = get_profile(profile_id)
        audio_path = synthesize(
            response_text,
            voice=profile["voice"],
            rate=profile["rate"],
            pitch=profile["pitch"],
        )

        facts = extract_facts_from_response(raw)
        if facts:
            save_session(PATIENT_ID, facts, "unknown", "unknown",
                         f"React session — {len(history)//2 + 1} turns")

        return response_text, audio_path or "", avatar_tag

    except Exception as e:
        print(f"[API Error] {e}")
        err_msg = "I'm sorry, I had a little trouble just now. Could you say that again?"
        profile = get_profile(profile_id)
        audio_path = synthesize(err_msg, voice=profile["voice"],
                                rate=profile["rate"], pitch=profile["pitch"])
        return err_msg, audio_path or "", "concerned"


# ---------------------------------------------------------------------------
# Gradio app — minimal UI; the real value is the /lumi_api predict endpoint
# ---------------------------------------------------------------------------

with gr.Blocks(title="Lumi API") as demo:
    gr.Markdown("## Lumi Backend API\nThis server powers the React frontend at port 80.")

    # Invisible components — register lumi_api as a callable predict endpoint
    _msg   = gr.Textbox(visible=False)
    _hist  = gr.JSON(visible=False)
    _prof  = gr.Textbox(visible=False)
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


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=["/tmp"],
        show_api=True,
    )

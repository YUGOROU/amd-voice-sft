---
title: Lumi — AI Voice Companion
emoji: 🌸
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
---

# Lumi — AI Voice Companion for Dementia & Alzheimer's Patients

Fine-tuned on AMD MI300X for the AMD Developer Hackathon 2026 (Fine-Tuning Track).

## What Lumi does

- **Warm, patient conversation** — remembers personal details across sessions (ChromaDB)
- **Scam protection** — detects and deflects elder fraud attempts without alarming the patient
- **Voice-native** — Whisper STT → Lumi → TTS, 21ms median time-to-first-audio
- **Clinically safe** — avoids clinical jargon; validated against EQ-Bench emotional intelligence benchmark

## Model

`Debdeep30/lumi-qwen3-4b-grpo` — Qwen3-4B fine-tuned via:
1. **SFT** — 8,540 dementia-care conversations, 3 epochs, QLoRA r=16
2. **GRPO Phase 1** — format compliance reward (100% by step 6)
3. **GRPO Phase 2** — format + length + clinical rewards

**EQ-Bench v2: 91.22/100** (base model: 0.00/100)

## Space secrets required

Set these in Space Settings → Variables and secrets:

| Secret | Value |
|--------|-------|
| `VLLM_BASE_URL` | Your vLLM endpoint, e.g. `http://YOUR_IP:8000/v1` |
| `MODEL_NAME` | `Debdeep30/lumi-qwen3-4b-grpo` |
| `PATIENT_NAME` | Patient's first name |
| `STT_URL` | Whisper-compatible endpoint (optional) |
| `TTS_URL` | `/v1/audio/speech`-compatible endpoint (optional) |

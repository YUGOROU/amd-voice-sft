---
title: Lumi - AI Dementia Companion
emoji: 🌿
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
tags:
  - amd
  - amd-hackathon-2026
  - dementia
  - voice-companion
  - fine-tuning
  - vllm
  - qwen3
---

# Lumi — AI Voice Companion for Dementia & Alzheimer's Patients

Fine-tuned Qwen3-4B served on AMD MI300X via vLLM.

## Space Secrets Required

| Secret | Value |
|---|---|
| `VLLM_BASE_URL` | `http://YOUR_DROPLET_IP:8000/v1` |
| `MODEL_NAME` | `YUGOROU/lumi-qwen3-4b` |
| `PATIENT_ID` | `demo_user_001` |
| `PATIENT_NAME` | `Margaret` |
| `STT_URL` | Whisper-compatible endpoint (optional) |
| `TTS_URL` | `/v1/audio/speech`-compatible endpoint (optional) |

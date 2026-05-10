---
title: Lumi Voice Companion
emoji: 🌿
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
tags:
  - amd
  - amd-hackathon-2026
  - vllm
  - gradio
  - voice-companion
  - dementia-care
---

# Lumi — AI Voice Companion

A warm AI companion for elderly users with memory difficulties, powered by AMD MI300X.

## Space Secrets

Configure these in **Settings → Variables and secrets** (add as Secrets, not Variables):

| Secret | Description | Example |
|--------|-------------|---------|
| `VLLM_BASE_URL` | LLM endpoint (vLLM on AMD) | `http://your-ip:8000/v1` |
| `MODEL_NAME` | Model ID loaded by vLLM | `meta-llama/Llama-3.1-8B-Instruct` |
| `STT_URL` | Granite Speech endpoint | `http://your-ip:8001` |
| `TTS_URL` | VoxCPM2 endpoint | `http://your-ip:8002` |

`STT_URL` and `TTS_URL` are optional — the app degrades gracefully to text-only if not set.

## Modes

- **Text Chat** — streaming text conversation with Lumi
- **Text + Voice Response** — text input, Lumi replies with synthesized speech
- **Voice Chat** — full voice pipeline: microphone → STT → LLM → TTS

## Backend

| Service | Model | Port |
|---------|-------|------|
| LLM | vLLM (configurable) | 8000 |
| STT | ibm-granite/granite-speech-4.1-2b-nar | 8001 |
| TTS | openbmb/VoxCPM2 (vLLM-Omni) | 8002 |

All three services run on AMD MI300X via ROCm.

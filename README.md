# amd-voice-sft

AI Voice Companion for elderly users with dementia and Alzheimer's — domain-specific SFT on AMD MI300X.

## Project Overview

This repository contains the data pipeline and training scripts for **Lumi**, an AI voice companion designed for elderly users with memory difficulties. The model is fine-tuned on domain-specific conversational data generated with EQ-Matrix parameters (condition, severity, emotion, scenario).

## Repository Structure

```
amd-voice-sft/
├── preprocess.ipynb      # Step 1: Raw dataset → ChatML format
├── crof_pipeline.ipynb   # Step 2: ChatML → EQ-Matrix domain data (crof.ai)
└── train_sft.py          # Step 3: SFT on AMD MI300X  [feature/sft-training only]
```

## Files

### `preprocess.ipynb`
**Purpose:** Converts 3 public HuggingFace datasets into ChatML format and uploads the result to HF Hub.

**Input datasets:**
| Dataset | Split | Rows | Type |
|---|---|---|---|
| `fadodr/mental_health_therapy` | train | 8,580 | Single-turn |
| `Estwld/empathetic_dialogues_llm` | train | 19,533 | Multi-turn |
| `HuggingFaceTB/everyday-conversations-llama3.1-2k` | train_sft | 2,260 | Multi-turn |

**Pipeline:**
1. Load each dataset and convert to `{"messages": [...]}` ChatML format
2. Inject a shared system prompt (Lumi persona) as the first message
3. Apply quality filter (user ≥ 20 chars, assistant ≥ 50 chars)
4. Shuffle and split 90:10 → `train.jsonl` / `val.jsonl`
5. Upload to `YUGOROU/amd-voice-sft-data` on HF Hub

**Output:** ~12,900 train + ~1,400 val samples at `YUGOROU/amd-voice-sft-data`

**Required Colab Secrets:** `HF_TOKEN`

---

### `crof_pipeline.ipynb`
**Purpose:** Rewrites ChatML samples into dementia-care domain data using the EQ-Matrix, then filters by quality score.

**EQ-Matrix parameters (2×3×5×5 = 150 combinations):**
- `condition`: dementia, alzheimer's
- `severity`: mild, moderate, severe
- `emotion`: calm, anxious, nostalgic, agitated, withdrawn
- `scenario`: repetitive_questions, time_place_confusion, family_memories, daily_care, social_interaction

**Pipeline:**
- **Layer 1 — Rewrite** (Cell 5): Each sample is rewritten by `deepseek-v4-flash` via crof.ai with randomly sampled EQ-Matrix parameters. Every assistant turn follows a strict 3-part format:
  ```
  [ACTION_TAG] first utterance (≤8 words)
  <think>
  patient state reasoning
  </think>
  final response (≤25 words, voice-optimized)
  ```
  Runs with `ThreadPoolExecutor(max_workers=20)` for parallelism. Output pushed to `YUGOROU/amd-voice-sft-dataset/rewritten/`.

- **Layer 2 — Filter** (Cell 6): Each rewritten sample is scored on 4 criteria (empathy, voice_suitability, domain_fit, format_compliance, each 0–10). Samples scoring ≥32/40 are kept. Uses `reasoning_effort="none"` and JSON schema enforcement for speed. Output pushed to `YUGOROU/amd-voice-sft-dataset/filtered/`.

**Required Colab Secrets:** `CROF_API_KEY`, `HF_TOKEN`

---

## Data Flow

```
HuggingFace (3 datasets)
        │
        ▼
preprocess.ipynb
        │  ChatML + quality filter
        ▼
YUGOROU/amd-voice-sft-data  (train.jsonl, val.jsonl)
        │
        ▼
crof_pipeline.ipynb
        │  Layer 1: EQ-Matrix rewrite
        ▼
YUGOROU/amd-voice-sft-dataset/rewritten/
        │  Layer 2: quality filter (≥32/40)
        ▼
YUGOROU/amd-voice-sft-dataset/filtered/
        │
        ▼
train_sft.py  [feature/sft-training]
        │  SFT on AMD MI300X
        ▼
YUGOROU/lumi-lora
```

## Environment

- **Preprocessing / Pipeline:** Google Colab (GPU not required)
- **Training:** AMD Dev Cloud — MI300X (192GB VRAM), ROCm 6.0+

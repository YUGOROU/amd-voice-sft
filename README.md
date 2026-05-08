# Lumi — AI Voice Companion for Dementia & Alzheimer's Patients

> AMD Developer Hackathon 2026 · Fine-Tuning Track · lablab.ai · HuggingFace Spaces · AMD MI300X

## What is Lumi?

Lumi is a fine-tuned AI voice companion designed specifically for elderly patients with dementia and Alzheimer's disease. It combines emotional intelligence, persistent cross-session memory, and a responsive voice interaction pipeline to provide companionship, cognitive stimulation, and protection from scams.

## What Makes Lumi Different

| Feature | Existing Apps | Lumi |
|---|---|---|
| Domain fine-tuned model | Generic LLMs | QLoRA on 8,500+ dementia-care samples |
| Persistent cross-session memory | No | ChromaDB, session facts |
| Structured latency-hiding output | No | Opening line fires TTS instantly |
| Scam deflection layer | No | Keyword + embedding classifier |
| Family dashboard | No | Session summaries + instructions |
| AMD GPU native | No | ROCm, vLLM, MI300X |

## Repository Structure

```
├── training/
│   ├── train_sft.py           # QLoRA fine-tuning script (Qwen3-4B, ROCm)
│   └── config/
│       ├── lora_config.yaml
│       └── training_args.yaml
├── pipeline/
│   ├── parser.py              # Structured output parser (avatar tag + think block)
│   ├── memory.py              # ChromaDB session memory
│   ├── stt.py                 # Whisper STT wrapper
│   ├── tts.py                 # Piper/Coqui TTS wrapper
│   └── scam_filter.py         # Scam detection classifier
├── demo/
│   ├── app.py                 # Gradio HF Space app (voice + avatar + memory + dashboard)
│   ├── requirements.txt
│   ├── README.md              # HF Space YAML config
│   └── avatar/                # 5 PNG expression images (smile/nod/concerned/gentle/laugh)
├── eval/
│   ├── run_rouge.py           # ROUGE-L evaluation
│   ├── latency_bench.py       # Time-to-first-audio benchmark (target <1.5s)
│   ├── structured_output_compliance.py  # Format compliance check (target >95%)
│   └── scam_eval.py           # Scam detection F1 (target >0.85)
└── docs/
    └── architecture.png       # System architecture diagram
```

## Data Pipeline

The dataset (`YUGOROU/lumi-data`) was built in 3 stages:

1. **Preprocess** — 3 public HF datasets converted to ChatML format
2. **EQ-Matrix rewrite** (crof.ai) — Layer 1 domain rewrite using 150 patient profile combinations
3. **Quality filter** — Layer 2 holistic scoring, 8,540 samples retained (69% keep rate)

Every assistant turn follows the structured output format:
```
[avatar_tag] Short opening line.
<think>
Internal reasoning — never sent to TTS or shown to user.
</think>
Full warm companion response.
```

## Training

```bash
export HF_TOKEN=your_token
python training/train_sft.py
```

Model: **Qwen3-4B-Instruct** (QLoRA r=16, 3 epochs, fp16, AMD MI300X)
Dataset: `YUGOROU/lumi-data` config `filtered`

ROCm notes:
- `HSA_OVERRIDE_GFX_VERSION=9.4.2` is set automatically
- Always `fp16=True`, NOT `bf16` — bf16 has incomplete ROCm support

## Serving

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./lumi-qwen3-output \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --max-model-len 4096
```

## Evaluation

```bash
# Structured output compliance
python eval/structured_output_compliance.py --model YUGOROU/lumi-qwen3-4b

# ROUGE-L
python eval/run_rouge.py --model YUGOROU/lumi-qwen3-4b --references eval/references.jsonl

# Latency benchmark
python eval/latency_bench.py --model YUGOROU/lumi-qwen3-4b

# Scam detection F1
python eval/scam_eval.py
```

## Evaluation Targets

| Metric | Target | Tool |
|---|---|---|
| ROUGE-L | > 0.35 | HuggingFace evaluate |
| Structured output compliance | > 95% | Custom regex |
| Time-to-first-audio | < 1.5s | Python time |
| Scam detection F1 | > 0.85 | sklearn |

## Avatar Setup

Generate 5 expression variants (smile, nod, concerned, gentle, laugh) using Midjourney or DALL-E 3:
> "Soft watercolour illustration of a gentle elderly female companion, warm eyes, no text, transparent background, front-facing portrait"

Save as `demo/avatar/avatar_{tag}.png` for each of the 5 tags.

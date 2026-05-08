"""
EQ-Bench v2 evaluation for model comparison.

Usage:
  python eval_eqbench.py \
    --models YUGOROU/lumi-qwen35-4b-test YUGOROU/lumi-gemma4-31b-test \
    --names  qwen35-4b gemma4-31b

Dataset: EQ-Bench/EQ-Bench (Apache-2.0)
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

HF_TOKEN = os.getenv("HF_TOKEN", "")

EQBENCH_REPO  = "EQ-Bench/EQ-Bench"
MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.01   # near-greedy for reproducibility


# ---------------------------------------------------------------------------
# EQ-Bench v2 scoring
# ---------------------------------------------------------------------------

def parse_emotion_scores(text: str) -> dict[str, float] | None:
    """Extract {emotion: score} from model output. Handles JSON or plain text."""
    # Try JSON block first
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group())
            return {k.lower().strip(): float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: "Emotion: N" lines
    scores = {}
    for line in text.splitlines():
        m = re.match(r'^\s*([A-Za-z ]+):\s*([0-9]+(?:\.[0-9]+)?)', line)
        if m:
            scores[m.group(1).lower().strip()] = float(m.group(2))
    return scores if scores else None


def eqbench_score(predicted: dict, reference: dict) -> float:
    """
    EQ-Bench v2 score for one question.
    score = (1 - mean(|pred_i - ref_i| / 10)) * 100
    """
    keys = [k for k in reference if k in predicted]
    if not keys:
        return 0.0
    deviations = [abs(predicted[k] - reference[k]) / 10.0 for k in keys]
    return (1.0 - sum(deviations) / len(deviations)) * 100.0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()
    return model, tokenizer


def run_eqbench(model, tokenizer, dataset) -> tuple[float, int]:
    scores, skipped = [], 0
    for item in dataset:
        messages = [
            {"role": "user", "content": item["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        reference = item["reference_answer"]
        if isinstance(reference, str):
            try:
                reference = json.loads(reference)
            except json.JSONDecodeError:
                skipped += 1
                continue
        reference = {k.lower().strip(): float(v) for k, v in reference.items()}

        predicted = parse_emotion_scores(response)
        if predicted is None:
            skipped += 1
            continue

        scores.append(eqbench_score(predicted, reference))

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="HF model IDs or local paths")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Short display names (defaults to model IDs)")
    args = parser.parse_args()

    names = args.names or args.models
    assert len(names) == len(args.models), "--names must match --models count"

    print("Loading EQ-Bench dataset ...")
    ds = load_dataset(EQBENCH_REPO, split="test", token=HF_TOKEN or None)
    print(f"  {len(ds):,} questions")

    results = {}
    for model_id, name in zip(args.models, names):
        model, tokenizer = load_model(model_id)
        score, skipped = run_eqbench(model, tokenizer, ds)
        results[name] = {"score": score, "skipped": skipped, "total": len(ds)}
        print(f"  [{name}] EQ-Bench: {score:.2f}  (skipped {skipped}/{len(ds)})")
        del model
        torch.cuda.empty_cache()

    print("\n=== EQ-Bench Results ===")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["score"]):
        print(f"  {name:30s}  {r['score']:.2f} / 100"
              f"  (skipped {r['skipped']}/{r['total']})")

    winner = max(results, key=lambda k: results[k]["score"])
    print(f"\nWinner: {winner} ({results[winner]['score']:.2f})")


if __name__ == "__main__":
    main()

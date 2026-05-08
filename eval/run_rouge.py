"""
ROUGE-L evaluation against a reference set.

Usage:
  python eval/run_rouge.py \
    --model YUGOROU/lumi-qwen3-4b \
    --references eval/references.jsonl \
    --output eval/results/rouge.json

references.jsonl format:
  {"input": "...", "reference": "..."}
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_references(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_responses(model_name: str, inputs: list[str], base_url: str) -> list[str]:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="not-required")
    responses = []
    for i, text in enumerate(inputs):
        print(f"  [{i+1}/{len(inputs)}] Generating ...", end="\r")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are Lumi, a warm companion for elderly patients."},
                {"role": "user",   "content": text},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        # strip structured output format for ROUGE comparison
        from pipeline.parser import parse_structured_output
        parsed = parse_structured_output(raw)
        responses.append(parsed["full_response"])
    print()
    return responses


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    from evaluate import load as eval_load
    rouge = eval_load("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return {k: round(v, 4) for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--base-url",   default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--output",     default="eval/results/rouge.json")
    args = parser.parse_args()

    refs = load_references(args.references)
    inputs     = [r["input"]     for r in refs]
    references = [r["reference"] for r in refs]

    print(f"Generating {len(inputs)} responses from {args.model} ...")
    predictions = generate_responses(args.model, inputs, args.base_url)

    print("Computing ROUGE-L ...")
    scores = compute_rouge(predictions, references)
    print(f"Results: {scores}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"model": args.model, "n_samples": len(refs), **scores}, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

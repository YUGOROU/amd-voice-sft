"""
Checks what % of model outputs conform to the structured output format.

Target: > 95%

Usage:
  python eval/structured_output_compliance.py \
    --model YUGOROU/lumi-qwen3-4b \
    --n 100 \
    --output eval/results/compliance.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.parser import VALID_TAGS, _TAG_RE, _THINK_RE

TEST_INPUTS = [
    "Good morning! How are you today?",
    "I keep forgetting where I put my glasses.",
    "My daughter visited, I think it was recently.",
    "I feel a bit sad today.",
    "What day is it today?",
    "Tell me about the garden.",
    "I miss my husband terribly.",
    "Can you remind me who I am?",
    "Someone called and said I won a prize.",
    "I'm confused about where I am.",
]


def check_compliance(text: str) -> dict:
    has_tag       = bool(_TAG_RE.match(text.strip()))
    valid_tag     = False
    has_think     = "<think>" in text and "</think>" in text
    has_response  = False

    if has_tag:
        m = _TAG_RE.match(text.strip())
        valid_tag = m.group(1) in VALID_TAGS

    if has_think:
        after_think = _THINK_RE.sub("", text).strip()
        tag_removed = _TAG_RE.sub("", after_think).strip()
        has_response = len(tag_removed) > 10

    return {
        "has_tag":      has_tag,
        "valid_tag":    valid_tag,
        "has_think":    has_think,
        "has_response": has_response,
        "fully_compliant": has_tag and valid_tag and has_think and has_response,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--n",        type=int, default=100)
    parser.add_argument("--output",   default="eval/results/compliance.json")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="not-required")

    results = []
    for i in range(args.n):
        prompt = TEST_INPUTS[i % len(TEST_INPUTS)]
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are Lumi, a warm companion for elderly patients."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=256,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content or ""
        result = check_compliance(raw)
        result["prompt"] = prompt
        results.append(result)
        print(f"  [{i+1:>3}/{args.n}] compliant={result['fully_compliant']} tag={result['valid_tag']}", end="\r")

    print()
    n_compliant = sum(r["fully_compliant"] for r in results)
    compliance_rate = n_compliant / len(results)

    summary = {
        "model":           args.model,
        "n":               args.n,
        "n_compliant":     n_compliant,
        "compliance_rate": round(compliance_rate, 4),
        "target_met":      compliance_rate >= 0.95,
        "has_tag_rate":    round(sum(r["has_tag"]      for r in results) / len(results), 4),
        "has_think_rate":  round(sum(r["has_think"]    for r in results) / len(results), 4),
    }
    print(f"\n{summary}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

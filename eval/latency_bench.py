"""
Time-to-first-audio latency benchmark.

Measures the key UX metric: time from when the user finishes speaking to
when TTS audio starts playing (i.e. how long until the opening line fires).

Usage:
  python eval/latency_bench.py \
    --model YUGOROU/lumi-qwen3-4b \
    --n 20 \
    --output eval/results/latency.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_INPUTS = [
    "Good morning! How are you today?",
    "I keep forgetting where I put my glasses.",
    "My daughter visited yesterday, I think.",
    "Can you sing me a song?",
    "I feel a bit sad today.",
    "What day is it today?",
    "Tell me about the garden.",
    "I miss my husband.",
    "Do you remember what we talked about last time?",
    "I don't know where I am.",
]


def measure_latencies(model_name: str, base_url: str, n: int) -> list[float]:
    from openai import OpenAI
    from pipeline.parser import parse_structured_output

    client = OpenAI(base_url=base_url, api_key="not-required")
    latencies = []

    for i in range(n):
        prompt = TEST_INPUTS[i % len(TEST_INPUTS)]
        messages = [
            {"role": "system", "content": "You are Lumi, a warm companion for elderly patients."},
            {"role": "user",   "content": prompt},
        ]

        t_start = time.perf_counter()
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.7,
            stream=True,
        )

        buffer = ""
        opening_fired = False
        t_opening = None

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            buffer += delta

            if not opening_fired and "</think>" in buffer:
                # Qwen3 outputs <think>...</think> first, then [tag] opening line.
                # TTS fires as soon as </think> is seen — the opening line is next.
                t_opening = time.perf_counter()
                opening_fired = True

        if t_opening is None:
            # fallback: first token with content
            t_opening = time.perf_counter()

        latency = t_opening - t_start
        latencies.append(latency)
        print(f"  [{i+1:>2}/{n}] {prompt[:40]:<40} → {latency:.3f}s")

    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--n",        type=int, default=20)
    parser.add_argument("--output",   default="eval/results/latency.json")
    args = parser.parse_args()

    print(f"Benchmarking {args.model} ({args.n} requests) ...")
    latencies = measure_latencies(args.model, args.base_url, args.n)

    results = {
        "model":   args.model,
        "n":       args.n,
        "mean_s":  round(statistics.mean(latencies), 3),
        "median_s": round(statistics.median(latencies), 3),
        "p90_s":   round(sorted(latencies)[int(0.9 * len(latencies))], 3),
        "min_s":   round(min(latencies), 3),
        "max_s":   round(max(latencies), 3),
        "target_met": statistics.mean(latencies) < 1.5,
    }

    print(f"\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

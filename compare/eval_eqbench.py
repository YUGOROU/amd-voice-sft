"""
EQ-Bench 3 evaluation wrapper for model comparison.

Runs eqbench3 for each vLLM-hosted model using deepseek-v4-flash as judge
via crof.ai, then prints a comparison table.

Setup:
  git clone https://github.com/EQ-bench/eqbench3 eqbench3_repo
  pip install -r eqbench3_repo/requirements.txt

Usage:
  CROF_API_KEY=xxx python eval_eqbench.py \
    --models  YUGOROU/lumi-qwen35-4b-test  YUGOROU/lumi-gemma4-31b-test \
    --names   qwen35-4b                    gemma4-31b \
    --vllm-urls http://localhost:8000      http://localhost:8001

Each --vllm-url is the base URL of a running vLLM server for that model
(e.g. `vllm serve YUGOROU/lumi-qwen35-4b-test --port 8000`).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EQBENCH3_DIR  = Path(__file__).parent / "eqbench3_repo"
JUDGE_MODEL   = "deepseek-v4-flash"
JUDGE_API_URL = "https://crof.ai/v1/chat/completions"


def clone_eqbench3():
    if not EQBENCH3_DIR.exists():
        print("Cloning EQ-Bench 3 ...")
        subprocess.run(
            ["git", "clone", "https://github.com/EQ-bench/eqbench3",
             str(EQBENCH3_DIR)],
            check=True,
        )
    else:
        print(f"Using existing {EQBENCH3_DIR}")


def run_eqbench(model_name: str, vllm_url: str, crof_api_key: str,
                runs_file: Path, elo_file: Path, iterations: int = 1):
    """Run eqbench3.py for one model and return the rubric score."""
    env = os.environ.copy()
    env.update({
        # Judge — deepseek-v4-flash via crof.ai
        "JUDGE_API_KEY": crof_api_key,
        "JUDGE_API_URL": JUDGE_API_URL,
        # Test model — transformers-based OpenAI-compatible endpoint
        # eqbench3 reads TEST_API_KEY / TEST_API_URL (full endpoint URL)
        "TEST_API_KEY": "vllm",
        "TEST_API_URL": vllm_url.rstrip("/") + "/v1/chat/completions",
    })

    # api_model_id must match the ID reported by the server's /v1/models endpoint
    # (serve.py uses adapter.split("/")[-1] as the model name)
    api_model_id = model_name
    cmd = [
        sys.executable,
        str(EQBENCH3_DIR / "eqbench3.py"),
        "--test-model",    f"openai/{api_model_id}",
        "--judge-model",   f"openai/{JUDGE_MODEL}",
        "--model-name",    model_name,
        "--iterations",    str(iterations),
        "--runs-file",     str(runs_file),
        "--elo-results-file", str(elo_file),
        "--threads",       "1",         # serialize requests to local server
        "--ignore-canonical",
        "--no-elo",
    ]

    print(f"\nRunning EQ-Bench 3 for {model_name} ...")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=EQBENCH3_DIR)


def extract_rubric_score(runs_file: Path, model_name: str) -> float | None:
    """Parse the runs JSON and compute mean rubric score for the model."""
    if not runs_file.exists():
        return None
    with open(runs_file) as f:
        data = json.load(f)

    scores = []
    for run in data.get("runs", []):
        if run.get("model_name") != model_name:
            continue
        rubric = run.get("rubric_scores", {})
        for scenario_scores in rubric.values():
            if isinstance(scenario_scores, (int, float)):
                scores.append(float(scenario_scores))
            elif isinstance(scenario_scores, list):
                scores.extend(float(s) for s in scenario_scores if s is not None)

    return sum(scores) / len(scores) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",    nargs="+", required=True,
                        help="HF model IDs (used as model names for vLLM)")
    parser.add_argument("--names",     nargs="+", default=None,
                        help="Short display names (defaults to --models)")
    parser.add_argument("--vllm-urls", nargs="+", required=True,
                        help="vLLM base URLs for each model in order")
    parser.add_argument("--iterations", type=int, default=1,
                        help="EQ-Bench iterations per scenario (default 1)")
    args = parser.parse_args()

    names = args.names or args.models
    assert len(names) == len(args.models) == len(args.vllm_urls), \
        "--models, --names, --vllm-urls must all have the same length"

    crof_api_key = os.getenv("CROF_API_KEY", "")
    assert crof_api_key, "Set CROF_API_KEY environment variable."

    clone_eqbench3()
    results = {}

    for model_id, name, vllm_url in zip(args.models, names, args.vllm_urls):
        runs_file = EQBENCH3_DIR / f"runs_{name}.json"
        elo_file  = EQBENCH3_DIR / f"elo_{name}.json"

        run_eqbench(
            model_name=name,
            vllm_url=vllm_url,
            crof_api_key=crof_api_key,
            runs_file=runs_file,
            elo_file=elo_file,
            iterations=args.iterations,
        )

        score = extract_rubric_score(runs_file, name)
        results[name] = score
        print(f"  [{name}] rubric score: {score:.2f}" if score else
              f"  [{name}] score not found in output")

    print("\n=== EQ-Bench 3 Results ===")
    for name, score in sorted(results.items(),
                               key=lambda x: x[1] or -1, reverse=True):
        label = f"{score:.2f}" if score is not None else "N/A"
        print(f"  {name:35s}  {label}")

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        winner = max(valid, key=lambda k: valid[k])
        print(f"\nWinner: {winner} ({valid[winner]:.2f})")
        print("Next step: run full training with the winner's script.")


if __name__ == "__main__":
    main()

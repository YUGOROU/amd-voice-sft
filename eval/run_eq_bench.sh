#!/usr/bin/env bash
# ==============================================================
# EQ-Bench 3 – Rubric pass for YUGOROU/lumi-grpo
#
# Prerequisites:
#   - Terminal 1: vLLM serving lumi-grpo on port 8000
#   - CROF_API_KEY: judge model API key (crof.ai)
#
# Usage:
#   # Terminal 1 – start vLLM
#   HSA_OVERRIDE_GFX_VERSION=9.4.2 HF_TOKEN=xxx \
#     python -m vllm.entrypoints.openai.api_server \
#     --model YUGOROU/lumi-grpo --dtype bfloat16 --device rocm \
#     --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.35
#
#   # Terminal 2 – run benchmark
#   CROF_API_KEY=xxx bash eval/run_eq_bench.sh
# ==============================================================
set -euo pipefail

: "${CROF_API_KEY:?Set CROF_API_KEY (used as judge model via crof.ai)}"

TEST_API_URL="${TEST_API_URL:-http://localhost:8000}"
TEST_MODEL="${TEST_MODEL:-YUGOROU/lumi-grpo}"
MODEL_NAME="${MODEL_NAME:-lumi-grpo}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
THREADS="${THREADS:-1}"
EQ_DIR="./eqbench3"

LOG() { echo "[$(date +'%H:%M:%S')] $*"; }

# ── 1. Clone eqbench3 ─────────────────────────────────────
if [ ! -d "$EQ_DIR" ]; then
    LOG "Cloning EQ-Bench 3..."
    git clone --depth 1 https://github.com/EQ-bench/eqbench3 "$EQ_DIR"
fi
cd "$EQ_DIR"

# ── 2. Install dependencies ───────────────────────────────
LOG "Installing requirements..."
pip install -q -r requirements.txt

# ── 3. Write .env ─────────────────────────────────────────
cat > .env << ENV
# Test model (lumi-grpo via vLLM)
TEST_API_URL=${TEST_API_URL}
TEST_API_KEY=EMPTY

# Judge model (crof.ai – deepseek)
JUDGE_API_URL=https://crof.ai/v1
JUDGE_API_KEY=${CROF_API_KEY}

REQUEST_TIMEOUT=300
MAX_RETRIES=3
LOG_VERBOSITY=INFO
ENV

LOG ".env written."

# ── 4. Verify vLLM is reachable ───────────────────────────
LOG "Checking vLLM at ${TEST_API_URL%/v1}/health ..."
if ! curl -sf "${TEST_API_URL%/v1}/health" >/dev/null 2>&1; then
    echo "ERROR: vLLM not responding at $TEST_API_URL"
    echo "Start it first:"
    echo "  HSA_OVERRIDE_GFX_VERSION=9.4.2 HF_TOKEN=xxx \\"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $TEST_MODEL --dtype bfloat16 --device rocm \\"
    echo "    --port 8000 --max-model-len 4096"
    exit 1
fi
LOG "vLLM ready."

# ── 5. Run EQ-Bench 3 (Rubric pass) ──────────────────────
LOG "Running EQ-Bench 3 Rubric pass..."
LOG "  Test model  : $TEST_MODEL"
LOG "  Judge model : $JUDGE_MODEL"
LOG "  Threads     : $THREADS"

python eqbench3.py \
    --test-model  "$TEST_MODEL" \
    --model-name  "$MODEL_NAME" \
    --judge-model "$JUDGE_MODEL" \
    --threads     "$THREADS"

LOG "EQ-Bench 3 complete. Results in $EQ_DIR/results/"

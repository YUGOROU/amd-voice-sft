#!/usr/bin/env bash
# ==============================================================
# Lumi Voice Companion – AMD MI300X Hosting Setup
#
# Usage:
#   HF_TOKEN=xxx bash <(curl -fsSL https://gist.github.com/YUGOROU/xxxx/raw/start.sh)
#
# Optional overrides:
#   LLM_MODEL=YUGOROU/lumi-grpo  (default)
#   WORKSPACE=/workspace/lumi-serve  (default)
# ==============================================================
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required. Usage: HF_TOKEN=xxx bash <(curl ...)}"

# ── AMD ROCm settings ──────────────────────────────────────
export HSA_OVERRIDE_GFX_VERSION="9.4.2"
export HF_HUB_DISABLE_XET="1"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN

# ── Config ─────────────────────────────────────────────────
LLM_MODEL="${LLM_MODEL:-YUGOROU/lumi-grpo}"
STT_MODEL="ibm-granite/granite-speech-4.1-2b"
LLM_PORT=8000
STT_PORT=8001
TTS_PORT=8002
API_PORT=8080

WORKSPACE="${WORKSPACE:-/workspace/lumi-serve}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

LOG() { echo "[$(date +'%H:%M:%S')] $*"; }

# ── 0. Firewall: expose only gateway port ─────────────────
# Internal services (8000-8002) stay on localhost; only 8080 is public
if command -v ufw &>/dev/null; then
    LOG "Configuring ufw: allow port $API_PORT only..."
    ufw allow "$API_PORT" comment "Lumi gateway" 2>/dev/null || true
    # Ensure internal vLLM ports are not exposed
    ufw deny 8000 2>/dev/null || true
    ufw deny 8001 2>/dev/null || true
    ufw deny 8002 2>/dev/null || true
    ufw --force enable 2>/dev/null || true
    LOG "Firewall configured. Only port $API_PORT is externally accessible."
else
    LOG "WARN: ufw not found, skipping firewall setup."
fi

# ── 1. Python environment (uv avoids Debian system-package conflicts) ────
if ! command -v uv &>/dev/null; then
    LOG "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

VENV="$WORKSPACE/venv"
if [ ! -f "$VENV/bin/python" ]; then
    LOG "Creating virtual environment at $VENV..."
    uv venv "$VENV"
fi
PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"

LOG "Installing Python dependencies..."
# Install ROCm torch
uv pip install --python "$PYTHON" -q torch \
    --index-url https://download.pytorch.org/whl/rocm6.3
uv pip install --python "$PYTHON" -q \
    "numpy>=1.26" "fastapi>=0.115" "uvicorn[standard]" httpx python-multipart \
    soundfile "transformers>=5.0" accelerate
# Install chatterbox-tts without pinned deps (torch/numpy version conflicts)
uv pip install --python "$PYTHON" -q --no-deps chatterbox-tts
uv pip install --python "$PYTHON" -q \
    conformer diffusers librosa omegaconf torchaudio s3tokenizer resemble-perth

# ── 2. Write TTS microservice ──────────────────────────────
cat > "$WORKSPACE/tts_server.py" << 'PYEOF'
import io
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

print("[TTS] Loading Chatterbox model (first run may download weights)...")
from chatterbox.tts import ChatterboxTTS
tts = ChatterboxTTS.from_pretrained(device="cuda")
print("[TTS] Ready.")


class SpeakReq(BaseModel):
    input: str
    voice: str = "default"
    model: str = "chatterbox"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/audio/speech")
async def speech(req: SpeakReq):
    wav = tts.generate(req.input)
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze().cpu().float().numpy(), 22050, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
PYEOF

# ── 3. Write FastAPI gateway ──────────────────────────────
cat > "$WORKSPACE/gateway.py" << PYEOF
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Lumi Voice Gateway")

LLM_BASE  = "http://localhost:${LLM_PORT}/v1"
STT_BASE  = "http://localhost:${STT_PORT}/v1"
TTS_BASE  = "http://localhost:${TTS_PORT}"
LLM_MODEL = "${LLM_MODEL}"
STT_MODEL = "${STT_MODEL}"


class TranscribeReq(BaseModel):
    audio: str  # base64-encoded WAV/MP3

class ChatReq(BaseModel):
    messages: List[Dict]
    stream: bool = False

class SpeakReq(BaseModel):
    text: str
    voice: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", "llm": LLM_BASE, "stt": STT_BASE, "tts": TTS_BASE}


@app.post("/transcribe")
async def transcribe(req: TranscribeReq):
    # Granite Speech 4.1 2B via vLLM multimodal chat completions
    payload = {
        "model": STT_MODEL,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{req.audio}"},
            }]
        }],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(f"{STT_BASE}/chat/completions", json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"STT error: {r.text}")
    return {"text": r.json()["choices"][0]["message"]["content"].strip()}


@app.post("/chat")
async def chat(req: ChatReq):
    payload = {
        "model": LLM_MODEL,
        "messages": req.messages,
        "stream": req.stream,
        "max_tokens": 256,
        "temperature": 0.7,
    }
    if req.stream:
        async def gen():
            async with httpx.AsyncClient(timeout=60.0) as c:
                async with c.stream("POST", f"{LLM_BASE}/chat/completions", json=payload) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(f"{LLM_BASE}/chat/completions", json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"LLM error: {r.text}")
    return {"response": r.json()["choices"][0]["message"]["content"]}


@app.post("/speak")
async def speak(req: SpeakReq):
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(f"{TTS_BASE}/v1/audio/speech",
                         json={"input": req.text, "voice": req.voice, "model": "chatterbox"})
    if r.status_code != 200:
        raise HTTPException(502, f"TTS error: {r.text}")
    return Response(content=r.content, media_type="audio/wav")


# OpenAI-compatible passthrough — allows HF Space backend_api.py to use VLLM_BASE_URL=http://this-host:8080
from fastapi import Request as FRequest
from fastapi.responses import JSONResponse as FJSONResponse

@app.get("/v1/models")
async def v1_models():
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{LLM_BASE}/models")
    return FJSONResponse(content=r.json())

@app.post("/v1/chat/completions")
async def v1_chat_completions(req: FRequest):
    body = await req.json()
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"{LLM_BASE}/chat/completions", json=body)
    return FJSONResponse(content=r.json(), status_code=r.status_code)
PYEOF

# ── 3b. Write LLM microservice (transformers-based, bypasses vLLM layer_scalar issue) ──
cat > "$WORKSPACE/llm_server.py" << 'PYEOF'
import argparse, time, uuid, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="YUGOROU/lumi-grpo")
parser.add_argument("--port",   type=int, default=8000)
parser.add_argument("--host",   default="0.0.0.0")
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN or None)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[LLM] Loading model on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16,
    device_map={"": device}, token=HF_TOKEN or None,
)
model.eval()
print(f"[LLM] Ready on {device}.")

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = args.model
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.get("/health")
@app.get("/")
@app.get("/v1")
def health():
    return {"status": "ok"}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": args.model, "object": "model"}]}

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/")
def chat_completions(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    max_new = req.max_tokens or args.max_new_tokens
    temp = req.temperature if req.temperature and req.temperature > 0 else None
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=max_new, do_sample=(temp is not None and temp > 0),
            temperature=temp or 1.0, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs.input_ids.shape[-1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion",
        "created": int(time.time()), "model": args.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": inputs.input_ids.shape[-1], "completion_tokens": len(new_tokens),
                  "total_tokens": inputs.input_ids.shape[-1] + len(new_tokens)},
    }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
PYEOF

# ── 4. Start background services ──────────────────────────
LOG "Starting LLM server (port $LLM_PORT) → $WORKSPACE/llm.log"
HF_TOKEN="$HF_TOKEN" "$PYTHON" "$WORKSPACE/llm_server.py" \
    --model "$LLM_MODEL" \
    --port "$LLM_PORT" \
    --host 127.0.0.1 \
    > "$WORKSPACE/llm.log" 2>&1 &
LLM_PID=$!

# STT skipped — vLLM not available on this image; /transcribe endpoint returns stub
STT_PID=0
echo "STT disabled" > "$WORKSPACE/stt.log"

LOG "Starting TTS server (port $TTS_PORT) → $WORKSPACE/tts.log"
"$VENV/bin/uvicorn" tts_server:app --host 127.0.0.1 --port "$TTS_PORT" \
    > "$WORKSPACE/tts.log" 2>&1 &
TTS_PID=$!

# ── 5. Graceful shutdown ──────────────────────────────────
cleanup() {
    LOG "Shutting down services (LLM=$LLM_PID STT=$STT_PID TTS=$TTS_PID)..."
    kill "$LLM_PID" "$STT_PID" "$TTS_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

# ── 6. Health check loop ──────────────────────────────────
wait_for() {
    local name="$1" url="$2" max="${3:-60}"
    printf "[%s] Waiting for %s" "$(date +'%H:%M:%S')" "$name"
    for _ in $(seq 1 "$max"); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo " ready ✓"
            return 0
        fi
        printf "."
        sleep 5
    done
    echo " TIMEOUT ✗"
    LOG "ERROR: $name did not start. Logs: $WORKSPACE/${name,,}.log"
    return 1
}

wait_for "LLM" "http://localhost:$LLM_PORT/health" 120  # up to 10 min (model download)
wait_for "TTS" "http://localhost:$TTS_PORT/health"  60

LOG "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
LOG "All services ready. Starting gateway on :$API_PORT"
LOG "  POST http://0.0.0.0:$API_PORT/transcribe  {audio: base64}"
LOG "  POST http://0.0.0.0:$API_PORT/chat        {messages, stream}"
LOG "  POST http://0.0.0.0:$API_PORT/speak       {text, voice}"
LOG "  GET  http://0.0.0.0:$API_PORT/health"
LOG "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 7. FastAPI gateway (foreground – keeps script alive) ──
"$VENV/bin/uvicorn" gateway:app --host 0.0.0.0 --port "$API_PORT" --workers 1

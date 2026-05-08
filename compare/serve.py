"""
Minimal OpenAI-compatible chat server for EQ-Bench evaluation.
Loads a base model + LoRA adapter using transformers + peft (ROCm-friendly).

Usage:
  python compare/serve.py \
    --base-model Qwen/Qwen3.5-4B \
    --adapter    YUGOROU/lumi-qwen35-4b-test \
    --port       8000

  python compare/serve.py \
    --base-model google/gemma-4-31b-it \
    --adapter    YUGOROU/lumi-gemma4-31b-test \
    --port       8001
"""
import argparse
import json
import os
import threading
import time
import uuid

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI()
model = None
tokenizer = None
model_name = ""
_gen_lock = threading.Lock()  # serialize GPU inference


# ---------------------------------------------------------------------------
# OpenAI-compatible request/response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 4000
    stream: bool = False


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with _gen_lock, torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=min(req.max_tokens, 2048),
                temperature=req.temperature if req.temperature > 0 else 1.0,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "server_error"}})

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": new_ids.shape[0],
            "total_tokens": inputs.input_ids.shape[1] + new_ids.shape[0],
        },
    }


# ---------------------------------------------------------------------------
# Manual LoRA merge (handles Gemma4ClippableLinear and similar wrappers)
# ---------------------------------------------------------------------------

def _manual_lora_merge(base, adapter_repo: str, token: str = "") -> None:
    """Merge LoRA weights directly into model parameters using safetensors.

    Works around PEFT's module-type whitelist (e.g. Gemma4ClippableLinear) by
    locating weights via named_parameters() and applying the LoRA delta in-place.
    """
    from safetensors.torch import load_file as st_load
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(adapter_repo, token=token or None)

    with open(os.path.join(local_dir, "adapter_config.json")) as f:
        cfg = json.load(f)

    r     = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    scale = alpha / r

    # Load all safetensors shards in the adapter repo
    state: dict = {}
    for fname in sorted(os.listdir(local_dir)):
        if fname.endswith(".safetensors"):
            state.update(st_load(os.path.join(local_dir, fname), device="cpu"))

    # Flat parameter map for O(1) lookup
    param_dict = dict(base.named_parameters())

    merged = 0
    for a_key in [k for k in state if k.endswith("lora_A.weight")]:
        b_key = a_key.replace("lora_A.weight", "lora_B.weight")
        if b_key not in state:
            continue

        # Strip PEFT key prefix: "base_model.model." or "base_model."
        raw = a_key
        for pfx in ("base_model.model.", "base_model."):
            if raw.startswith(pfx):
                raw = raw[len(pfx):]
                break
        param_name = raw.replace(".lora_A.weight", "")

        # Try direct weight, then inner .linear weight (Gemma4ClippableLinear)
        W = None
        for suffix in (".weight", ".linear.weight"):
            candidate = param_name + suffix
            if candidate in param_dict:
                W = param_dict[candidate]
                break

        if W is None:
            continue

        lora_A = state[a_key].to(W.device, dtype=W.dtype)
        lora_B = state[b_key].to(W.device, dtype=W.dtype)
        W.data.add_((lora_B @ lora_A) * scale)
        merged += 1

    print(f"  Manually merged {merged} LoRA layers (alpha/r = {scale:.3f})")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_model(base_model: str, adapter: str | None):
    global model, tokenizer, model_name
    model_name = adapter.split("/")[-1] if adapter else base_model.split("/")[-1]

    print(f"Loading tokenizer: {base_model}")
    try:
        tokenizer = AutoProcessor.from_pretrained(base_model, token=HF_TOKEN, padding_side="left")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )

    if adapter:
        print(f"Loading LoRA adapter: {adapter}")
        try:
            # Standard PEFT path (works for most models)
            model = PeftModel.from_pretrained(base, adapter, token=HF_TOKEN)
            model = model.merge_and_unload()
        except (ValueError, KeyError):
            # Fallback: manual safetensors merge (handles Gemma4ClippableLinear etc.)
            print("PEFT load failed, trying manual LoRA merge ...")
            _manual_lora_merge(base, adapter, HF_TOKEN)
            model = base
    else:
        model = base

    model.eval()
    print(f"Ready on {next(model.parameters()).device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter",    default=None)
    parser.add_argument("--port",       type=int, default=8000)
    args = parser.parse_args()

    load_model(args.base_model, args.adapter)
    uvicorn.run(app, host="0.0.0.0", port=args.port)

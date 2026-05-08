"""
Minimal OpenAI-compatible chat server for EQ-Bench evaluation.

- Non-Gemma models (Qwen etc.): loaded via Unsloth for ROCm GPU support
- Gemma4: loaded via plain transformers (Unsloth patches Gemma4TextConfig
  in a way that breaks config validation, so we never import it for Gemma4)

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

# Must be set before any ROCm/torch import
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# NOTE: Do NOT import unsloth at module level.
# Unsloth patches Gemma4TextConfig.__getattr__ which breaks transformers
# config validation. Import inside load_model() only for non-Gemma models.

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI()
model = None
tokenizer = None
model_name = ""
_gen_lock = threading.Lock()  # serialize GPU inference (one request at a time)


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
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)

        with _gen_lock, torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=min(req.max_tokens, 2048),
                temperature=req.temperature if req.temperature > 0 else 1.0,
                do_sample=req.temperature > 0,
                pad_token_id=getattr(getattr(tokenizer, "tokenizer", tokenizer), "pad_token_id", None),
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
    """Merge LoRA weights directly into model parameters using safetensors."""
    from safetensors.torch import load_file as st_load
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(adapter_repo, token=token or None)

    with open(os.path.join(local_dir, "adapter_config.json")) as f:
        cfg = json.load(f)

    r     = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    scale = alpha / r

    state: dict = {}
    for fname in sorted(os.listdir(local_dir)):
        if fname.endswith(".safetensors"):
            state.update(st_load(os.path.join(local_dir, fname), device="cpu"))

    param_dict = dict(base.named_parameters())

    merged = 0
    for a_key in [k for k in state if k.endswith("lora_A.weight")]:
        b_key = a_key.replace("lora_A.weight", "lora_B.weight")
        if b_key not in state:
            continue

        raw = a_key
        for pfx in ("base_model.model.", "base_model."):
            if raw.startswith(pfx):
                raw = raw[len(pfx):]
                break
        param_name = raw.replace(".lora_A.weight", "")

        W = None
        for suffix in (".weight", ".linear.weight"):
            if (param_name + suffix) in param_dict:
                W = param_dict[param_name + suffix]
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
    from huggingface_hub import login

    model_name = adapter.split("/")[-1] if adapter else base_model.split("/")[-1]
    if HF_TOKEN:
        login(token=HF_TOKEN or None)

    is_gemma = "gemma" in base_model.lower()
    dtype = torch.bfloat16 if "qwen3" in base_model.lower() else torch.float16
    print(f"Loading model: {base_model}  dtype={dtype}")

    if is_gemma:
        # Pure transformers path — never import Unsloth for Gemma4.
        # Patch caching_allocator_warmup if ROCm doesn't expose cudart.
        try:
            torch.cuda.cudart()
        except Exception:
            import transformers.modeling_utils as _mu
            _mu.caching_allocator_warmup = lambda *a, **kw: None

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map={"": "cuda:0"},
            token=HF_TOKEN or None,
        )
        try:
            tok = AutoProcessor.from_pretrained(base_model, token=HF_TOKEN or None, padding_side="left")
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN or None)
    else:
        # Unsloth path for ROCm-compatible loading of non-Gemma models.
        from unsloth import FastLanguageModel
        base, _tok = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=8192,
            dtype=dtype,
            load_in_4bit=False,
            token=HF_TOKEN or None,
        )
        # Unsloth may return a VLProcessor; use AutoTokenizer for text-only inference
        try:
            tok = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN or None)
        except Exception:
            tok = _tok

    tokenizer = tok
    _inner = getattr(tokenizer, "tokenizer", tokenizer)
    if _inner.pad_token is None:
        _inner.pad_token = _inner.eos_token

    if adapter:
        print(f"Loading LoRA adapter: {adapter}")
        try:
            base = PeftModel.from_pretrained(base, adapter, token=HF_TOKEN or None)
            base = base.merge_and_unload()
        except (ValueError, KeyError):
            print("PEFT load failed, trying manual LoRA merge ...")
            _manual_lora_merge(base, adapter, HF_TOKEN)

    if not is_gemma:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(base)

    base.eval()
    model = base
    print(f"Ready on {next(model.parameters()).device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter",    default=None)
    parser.add_argument("--port",       type=int, default=8000)
    args = parser.parse_args()

    load_model(args.base_model, args.adapter)
    uvicorn.run(app, host="0.0.0.0", port=args.port)

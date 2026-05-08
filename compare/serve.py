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
import os
import time
import uuid

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI()
model = None
tokenizer = None
model_name = ""


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
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature if req.temperature > 0 else 1.0,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

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
# Startup
# ---------------------------------------------------------------------------

def load_model(base_model: str, adapter: str | None):
    global model, tokenizer, model_name
    model_name = adapter.split("/")[-1] if adapter else base_model.split("/")[-1]

    print(f"Loading tokenizer: {base_model}")
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
            # Standard PEFT path
            model = PeftModel.from_pretrained(base, adapter, token=HF_TOKEN)
            model = model.merge_and_unload()
        except (ValueError, KeyError):
            # Fallback: transformers built-in adapter loader (handles Gemma4 etc.)
            print("PEFT load failed, trying transformers load_adapter()")
            base.load_adapter(adapter, adapter_name="default", token=HF_TOKEN)
            base.merge_adapter("default")
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

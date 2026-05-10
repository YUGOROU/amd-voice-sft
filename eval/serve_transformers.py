"""
Minimal OpenAI-compatible inference server using transformers directly.
Bypasses vLLM weight loader issues with Gemma-4 layer_scalar parameters.

Usage (inside rocm container):
  HF_TOKEN=xxx python3 eval/serve_transformers.py \
    --model YUGOROU/lumi-grpo --port 8000
"""
import argparse, time, uuid, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="YUGOROU/lumi-grpo")
parser.add_argument("--port",   type=int, default=8000)
parser.add_argument("--host",   default="0.0.0.0")
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"Loading tokenizer: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN or None)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model: {args.model} on {device} (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
    token=HF_TOKEN or None,
)
model.eval()
print(f"Model loaded on {device}.")
print("Model loaded.")

app = FastAPI(title="Lumi Transformers Server")


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
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    max_new = req.max_tokens or args.max_new_tokens
    temp = req.temperature if req.temperature and req.temperature > 0 else None

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=(temp is not None and temp > 0),
            temperature=temp or 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": args.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_ids.shape[-1],
            "completion_tokens": len(new_tokens),
            "total_tokens": input_ids.shape[-1] + len(new_tokens),
        },
    }


if __name__ == "__main__":
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

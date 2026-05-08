"""
Merge LoRA adapters into the base model and push the full model to HuggingFace Hub.

Run this after train_sft.py completes. vLLM needs the merged weights — it can't
serve a PEFT adapter directly without the base model alongside it.

Usage:
  python training/merge_and_push.py \
    --adapter ./lumi-qwen3-output \
    --base Qwen/Qwen3-4B-Instruct \
    --output-repo YUGOROU/lumi-qwen3-4b
"""

import argparse
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

HF_TOKEN = os.getenv("HF_TOKEN", "")
assert HF_TOKEN, "Set HF_TOKEN in .env before running."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",     required=True,
                        help="Path to the saved PEFT adapter (e.g. ./lumi-qwen3-output)")
    parser.add_argument("--base",        default="Qwen/Qwen3-4B-Instruct",
                        help="Base model ID used during training")
    parser.add_argument("--output-repo", default="YUGOROU/lumi-qwen3-4b",
                        help="HuggingFace repo to push the merged model to")
    parser.add_argument("--private",     action="store_true", default=True)
    args = parser.parse_args()

    login(token=HF_TOKEN)

    print(f"Loading base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN,
    )

    print(f"Loading adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Pushing merged model to {args.output_repo}...")
    model.push_to_hub(args.output_repo, token=HF_TOKEN, private=args.private)
    tokenizer.push_to_hub(args.output_repo, token=HF_TOKEN, private=args.private)


    print(f"Done. Merged model available at: https://huggingface.co/{args.output_repo}")
    print(f"\nStart vLLM with:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"    --model {args.output_repo} \\")
    print(f"    --host 0.0.0.0 --port 8000 \\")
    print(f"    --dtype float16 \\")
    print(f"    --max-model-len 4096")


if __name__ == "__main__":
    main()

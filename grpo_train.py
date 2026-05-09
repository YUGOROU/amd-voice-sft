"""
GRPO training for Lumi (Gemma-4-31B).

Staged reward injection (TeenEmo insight):
  Stage 1 (ep 1-2): format_reward only
  Stage 2 (ep 3-4): + constitutional_reward
  Stage 3 (ep 5-6): + character_reward
  Stage 4 (ep 7-8): + eq_bench_reward

Run AFTER sft_train.py has pushed YUGOROU/lumi-sft to HF Hub.

Prerequisites:
  1. vLLM server running:
       python -m vllm.entrypoints.openai.api_server \
         --model YUGOROU/lumi-sft --dtype bfloat16 --port 8000 --device rocm
  2. export CROF_API_KEY=...  HF_TOKEN=...
"""
import os
import time
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import login

from reward_functions import STAGE_CONFIGS

SFT_MODEL_REPO = os.getenv("SFT_MODEL", "YUGOROU/lumi-sft")
HF_DATASET     = "YUGOROU/lumi-data"
DATA_CONFIG    = "filtered"
OUTPUT_REPO    = "YUGOROU/lumi-grpo"
HF_TOKEN       = os.getenv("HF_TOKEN", "")
VLLM_URL       = os.getenv("VLLM_URL", "http://localhost:8000")

assert HF_TOKEN,             "Set HF_TOKEN environment variable."
assert os.getenv("CROF_API_KEY"), "Set CROF_API_KEY environment variable."
login(token=HF_TOKEN)


def wait_for_vllm(url: str, timeout: int = 120):
    print(f"Waiting for vLLM server at {url} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                print("vLLM server ready.")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM server at {url} did not become ready in {timeout}s")


wait_for_vllm(VLLM_URL)

print(f"Loading SFT model: {SFT_MODEL_REPO}")
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_REPO,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    use_cache=False,
    token=HF_TOKEN,
)

# LoRA必須: 31B full fine-tuneはoptimizer statesで192GB超過する
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_REPO, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset(HF_DATASET, DATA_CONFIG, split="train", token=HF_TOKEN)
print(f"Dataset: {len(raw):,} samples")

# promptフィールドのみ残す（GRPOはpromptから生成）
def extract_prompt(ex):
    # messagesの最後のassistant発話を除いたturnをpromptにする
    turns = [m for m in ex["messages"] if m["role"] != "assistant"]
    return {
        "prompt": tokenizer.apply_chat_template(
            turns, tokenize=False, add_generation_prompt=True
        )
    }

dataset = raw.map(extract_prompt, remove_columns=raw.column_names)

# ────────────────────────────────────────────
# 段階的GRPO (Stage 1 → 4)
# ────────────────────────────────────────────
checkpoint_dir = "./lumi-grpo-stage"

for stage_idx, (reward_funcs, reward_weights, n_epochs, desc) in enumerate(STAGE_CONFIGS, 1):
    stage_output = f"{checkpoint_dir}{stage_idx}"
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  epochs={n_epochs}, weights={reward_weights}")
    print(f"{'='*60}")

    grpo_config = GRPOConfig(
        output_dir=stage_output,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        use_vllm=True,
        vllm_server_host="localhost",
        vllm_server_port=8000,
        deepspeed="ds_config_grpo.json",
        report_to="none",
        reward_weights=reward_weights,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
    )
    trainer.train()

    # 次ステージへモデルを引き継ぐ
    model = trainer.model
    print(f"Stage {stage_idx} complete → {stage_output}")

# 最終モデル保存・push
print("\nMerging LoRA and pushing final model ...")
final_dir = "./lumi-grpo"
merged = model.merge_and_unload()
merged.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

merged.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
print(f"Pushed final GRPO model to {OUTPUT_REPO}")

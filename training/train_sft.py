import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Must be set before any ROCm/torch import
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# =============================================================================
# CONFIG
# =============================================================================

# Qwen2.5-3B — strong instruction-following, best ROCm support for this size.
# Alternatives (set BASE_MODEL and adjust USE_LORA / BATCH_SIZE accordingly):
#   "Qwen/Qwen2.5-7B-Instruct"          — more capacity, ~14GB fp16
#   "unsloth/Meta-Llama-3.1-8B-Instruct" — Llama option, needs unsloth
BASE_MODEL  = "Qwen/Qwen3-4B-Instruct"
OUTPUT_REPO = "YUGOROU/lumi-qwen3-4b"
DATA_REPO   = "YUGOROU/lumi-data"
DATA_CONFIG = "filtered"           # load_dataset(DATA_REPO, DATA_CONFIG)
HF_TOKEN    = os.getenv("HF_TOKEN", "")

USE_LORA     = True
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 2048
BATCH_SIZE   = 4
GRAD_ACCUM   = 4      # effective batch = 16
LR           = 2e-4
EPOCHS       = 3
WARMUP_RATIO = 0.03

# =============================================================================

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # TrainingArguments,  <-- replaced by SFTConfig
)
from trl import SFTTrainer, SFTConfig

assert HF_TOKEN, "Set HF_TOKEN environment variable before running."

login(token=HF_TOKEN)

# --- Tokenizer & model -------------------------------------------------------
print(f"Loading {BASE_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,  # explicit fp16 — "auto" may pick bf16 on ROCm
    device_map="auto",
    token=HF_TOKEN,
)

# --- LoRA -------------------------------------------------------------------
if USE_LORA:
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# --- Dataset -----------------------------------------------------------------
raw = load_dataset(DATA_REPO, DATA_CONFIG, split="train", token=HF_TOKEN)
print(f"Loaded {len(raw):,} samples from {DATA_REPO}/{DATA_CONFIG}")


def format_sample(ex):
    return {
        "text": tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = raw.map(format_sample, remove_columns=raw.column_names)
print(f"Sample[:200]: {dataset[0]['text'][:200]}")

# --- Train ------------------------------------------------------------------
sft_config = SFTConfig(
    output_dir="./lumi-qwen3-output",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=False,
    fp16=True,            # ROCm: bf16 support is incomplete, use fp16
    logging_steps=10,
    save_steps=200,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_num_workers=4,
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

stats = trainer.train()
print(f"Training complete. Loss: {stats.training_loss:.4f}")

# --- Push to Hub ------------------------------------------------------------
model.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
print(f"Pushed to {OUTPUT_REPO}")

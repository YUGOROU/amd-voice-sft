import os

# Must be set before any ROCm/torch import — tells ROCm to treat MI300X as gfx942
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# =============================================================================
# CONFIG — edit this block to switch models / training strategy
# =============================================================================

# TODO: choose base model before running
#   Option A — Llama 3.3 70B + LoRA (best quality, ~164GB VRAM)
#     BASE_MODEL = "unsloth/Llama-3.3-70B-Instruct"
#   Option B — Llama 3.1 8B + full fine-tuning (~96GB VRAM, fast iteration)
#     BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
#     USE_LORA   = False
BASE_MODEL  = "TODO"
OUTPUT_REPO = "YUGOROU/lumi-lora"
DATA_REPO   = "YUGOROU/lumi-data"
HF_TOKEN    = os.getenv("HF_TOKEN", "")

USE_LORA     = True   # False = full fine-tuning (8B only)
LORA_R       = 16
LORA_ALPHA   = 16
MAX_SEQ_LEN  = 2048
BATCH_SIZE   = 1      # 70B LoRA: keep at 1
GRAD_ACCUM   = 8      # effective batch = 8
LR           = 2e-4
EPOCHS       = 1
WARMUP_RATIO = 0.05

# =============================================================================

from datasets import load_dataset
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel

assert BASE_MODEL != "TODO", "Set BASE_MODEL in the config block before running."

# --- Model -------------------------------------------------------------------
model, tokenizer = FastModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,   # bf16 — avoids bitsandbytes NaN bug on AMD
    token=HF_TOKEN,
)

if USE_LORA:
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

print(model.print_trainable_parameters())

# --- Dataset -----------------------------------------------------------------
login(token=HF_TOKEN)

raw = load_dataset(DATA_REPO, data_dir="filtered", split="train", token=HF_TOKEN)
print(f"Loaded {len(raw):,} filtered samples from {DATA_REPO}/filtered")


def format_sample(ex):
    return {
        "text": tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = raw.map(format_sample, remove_columns=raw.column_names)
print(f"Sample text[:200]: {dataset[0]['text'][:200]}")

# --- Train -------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        output_dir="./lumi-output",
        report_to="none",
    ),
)

stats = trainer.train()
print(f"Training complete. Loss: {stats.training_loss:.4f}")

# --- Push to Hub -------------------------------------------------------------
model.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
print(f"Pushed to {OUTPUT_REPO}")

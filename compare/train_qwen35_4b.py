import os

# Must be set before any ROCm/torch import
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# unsloth must be imported before trl/transformers/peft
from unsloth import FastLanguageModel

# =============================================================================
# CONFIG
# =============================================================================

BASE_MODEL  = "Qwen/Qwen3.5-4B"
OUTPUT_DIR  = "./output/qwen35-4b-lumi-test"
OUTPUT_REPO = "YUGOROU/lumi-qwen35-4b-test"
DATA_REPO   = "YUGOROU/lumi-data"
DATA_CONFIG = "filtered"
HF_TOKEN    = os.getenv("HF_TOKEN", "")

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 2048
BATCH_SIZE   = 4
GRAD_ACCUM   = 4       # effective batch = 16
LR           = 2e-4
EPOCHS       = 1       # comparison test — increase for full run
WARMUP_RATIO = 0.03

# =============================================================================

import torch
from datasets import load_dataset
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig

assert HF_TOKEN, "Set HF_TOKEN environment variable."
login(token=HF_TOKEN)

# --- Model -------------------------------------------------------------------
print(f"Loading {BASE_MODEL} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,   # Qwen3.5 GatedDeltaNet requires bfloat16
    load_in_4bit=False,     # avoid bitsandbytes NaN bug on AMD
    token=HF_TOKEN,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
model.print_trainable_parameters()

# --- Dataset -----------------------------------------------------------------
print(f"Loading {DATA_REPO} [{DATA_CONFIG}] ...")
raw = load_dataset(DATA_REPO, DATA_CONFIG, split="train", token=HF_TOKEN)
print(f"  {len(raw):,} samples")


def format_sample(ex):
    return {
        "text": tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = raw.map(format_sample, remove_columns=raw.column_names)

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
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

stats = trainer.train()
print(f"Loss: {stats.training_loss:.4f}")

# --- Save & push -------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

model.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
print(f"Pushed to {OUTPUT_REPO}")

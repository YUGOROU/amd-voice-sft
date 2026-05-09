import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import login

MODEL_ID    = "google/gemma-4-31b-it"
HF_DATASET  = "YUGOROU/lumi-data"
DATA_CONFIG = "filtered"
OUTPUT_DIR  = "./lumi-sft"
OUTPUT_REPO = "YUGOROU/lumi-sft"
HF_TOKEN    = os.getenv("HF_TOKEN", "")

assert HF_TOKEN, "Set HF_TOKEN environment variable."
login(token=HF_TOKEN)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    token=HF_TOKEN,
)
model.config.use_cache = False
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# messages列(list[dict])をapply_chat_templateで文字列に変換
raw = load_dataset(HF_DATASET, DATA_CONFIG, split="train", token=HF_TOKEN)
print(f"Dataset: {len(raw):,} samples")

def format_sample(ex):
    return {
        "text": tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = raw.map(format_sample, remove_columns=raw.column_names)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=100,
    deepspeed="ds_config_sft.json",
    max_seq_length=2048,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# LoRAをベースモデルにマージしてから保存
print("Merging LoRA weights ...")
merged = trainer.model.merge_and_unload()
merged.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

merged.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
print(f"Pushed merged SFT model to {OUTPUT_REPO}")

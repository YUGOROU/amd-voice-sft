import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import login

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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


def apply_lora_gemma4(model, lora_cfg):
    """
    Gemma4ClippableLinear はPEFTのホワイトリスト外なので、
    一時的に内側のnn.Linearを露出させてLoRAを適用し、
    完了後にラッパーを復元する。
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return get_peft_model(model, lora_cfg)

    # Step1: ClippableLinear → 内側のlinearに一時置換
    wrappers = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Gemma4ClippableLinear):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], mod.linear)
            wrappers[name] = (parent, parts[-1], mod)

    print(f"Temporarily unwrapped {len(wrappers)} Gemma4ClippableLinear modules")

    # Step2: LoRA適用（内側のnn.Linearに対して実行される）
    peft_model = get_peft_model(model, lora_cfg)

    # Step3: ラッパーを復元（内側はLoRA済みのモジュールになっている）
    for name, (parent, child_name, wrapper) in wrappers.items():
        wrapper.linear = getattr(parent, child_name)
        setattr(parent, child_name, wrapper)

    print(f"Restored {len(wrappers)} Gemma4ClippableLinear wrappers with LoRA inside")
    return peft_model


model = apply_lora_gemma4(model, lora_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

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
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train(resume_from_checkpoint=True)

# LoRAをベースモデルにマージしてから保存
print("Merging LoRA weights ...")
merged = trainer.model.merge_and_unload()
merged.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

merged.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
print(f"Pushed merged SFT model to {OUTPUT_REPO}")

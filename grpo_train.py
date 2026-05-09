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

# Must be set before any ROCm/torch import
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import login

from reward_functions import STAGE_CONFIGS


def apply_lora_gemma4(model, lora_cfg):
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return get_peft_model(model, lora_cfg)

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
    peft_model = get_peft_model(model, lora_cfg)

    for name, (parent, child_name, wrapper) in wrappers.items():
        wrapper.linear = getattr(parent, child_name)
        setattr(parent, child_name, wrapper)

    print(f"Restored {len(wrappers)} Gemma4ClippableLinear wrappers with LoRA inside")
    return peft_model

SFT_MODEL_REPO  = os.getenv("SFT_MODEL", "YUGOROU/lumi-sft")
HF_DATASET      = "YUGOROU/lumi-data"
DATA_CONFIG     = "filtered"
OUTPUT_REPO     = "YUGOROU/lumi-grpo"
HF_TOKEN        = os.getenv("HF_TOKEN", "")
# 途中再開: START_STAGE=2 RESUME_CHECKPOINT=./lumi-grpo-stage2/checkpoint-100
START_STAGE     = int(os.getenv("START_STAGE", "1"))
RESUME_CHECKPOINT = os.getenv("RESUME_CHECKPOINT", None)

assert HF_TOKEN,             "Set HF_TOKEN environment variable."
assert os.getenv("CROF_API_KEY"), "Set CROF_API_KEY environment variable."
login(token=HF_TOKEN)



# ROCm: transformers 5.8のスレッドローダーがcudartを呼ぶのを回避するため
# CPUにロードしてからGPUへ移動する（RAM 240GB あるので問題なし）
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **kw: None

import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **kw: None

base_source = RESUME_CHECKPOINT if RESUME_CHECKPOINT else SFT_MODEL_REPO
print(f"Loading base model: {base_source}")
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_REPO,
    attn_implementation="sdpa",
    dtype=torch.bfloat16,
    device_map="cpu",
    token=HF_TOKEN,
)
model.config.use_cache = False
print("Moving model to cuda:0 ...")
model = model.to("cuda:0")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = apply_lora_gemma4(model, lora_config)

if RESUME_CHECKPOINT:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file as st_load
    import glob
    print(f"Loading LoRA weights from checkpoint: {RESUME_CHECKPOINT}")
    ckpt_files = sorted(glob.glob(f"{RESUME_CHECKPOINT}/adapter_model*.safetensors"))
    state_dict = {}
    for f in ckpt_files:
        state_dict.update(st_load(f, device="cpu"))
    set_peft_model_state_dict(model, state_dict)
    print(f"Loaded {len(state_dict)} tensors from checkpoint")
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
# GRPO は全件不要。多様性を保ちつつ 2,000 件に絞る
dataset = dataset.shuffle(seed=42).select(range(2000))

# ────────────────────────────────────────────
# 段階的GRPO (Stage 1 → 4)
# ────────────────────────────────────────────
checkpoint_dir = "./lumi-grpo-stage"

for stage_idx, (reward_funcs, reward_weights, n_epochs, desc) in enumerate(STAGE_CONFIGS, 1):
    if stage_idx < START_STAGE:
        print(f"Skipping Stage {stage_idx} (START_STAGE={START_STAGE})")
        continue
    stage_output = f"{checkpoint_dir}{stage_idx}"
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  epochs={n_epochs}, weights={reward_weights}")
    print(f"{'='*60}")

    grpo_config = GRPOConfig(
        output_dir=stage_output,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_completion_length=128,
        bf16=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=25,
        use_vllm=False,
        num_generations=4,
        report_to="none",
        reward_weights=reward_weights,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )
    resume = RESUME_CHECKPOINT if (stage_idx == START_STAGE and RESUME_CHECKPOINT) else None
    trainer.train(resume_from_checkpoint=resume)

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

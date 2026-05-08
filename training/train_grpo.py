import os
import re
from dotenv import load_dotenv

load_dotenv()

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# =============================================================================
# CONFIG
# =============================================================================

BASE_MODEL  = "Qwen/Qwen3-4B-Instruct"
DATA_REPO   = "YUGOROU/lumi-data"
DATA_CONFIG = "filtered"
HF_TOKEN    = os.getenv("HF_TOKEN", "")

# TRAINING_PHASE controls which reward functions are active:
#   1 = format compliance ONLY — run this first until compliance > 95%
#   2 = all rewards — add after Phase 1 checkpoint is stable
TRAINING_PHASE = 1

assert HF_TOKEN, "Set HF_TOKEN environment variable in .env before running."
login(token=HF_TOKEN)

# =============================================================================

print(f"Loading tokenizer from {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# --- Dataset -----------------------------------------------------------------

raw = load_dataset(DATA_REPO, DATA_CONFIG, split="train", token=HF_TOKEN)


def prepare_prompts(ex):
    prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}


dataset = raw.map(prepare_prompts)
print(f"Loaded {len(dataset):,} prompts. Sample:\n{dataset[0]['prompt'][:300]}")


# =============================================================================
# Reward Functions
# =============================================================================

_TAG_RE    = re.compile(r"^\[(smile|nod|concerned|gentle|laugh)\]", re.IGNORECASE)
_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL)
_VALID_TAGS = {"smile", "nod", "concerned", "gentle", "laugh"}


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Graded reward (0.0–1.0) for full structural compliance.

    +0.25  valid lowercase avatar tag at the start
    +0.25  non-empty opening line between tag and <think>
    +0.25  <think>...</think> block present
    +0.25  non-empty full response after the think block

    Using graded partial credit gives the model a gradient signal even when
    it's only partially compliant — binary rewards stall early training.
    """
    rewards = []
    for completion in completions:
        text = completion.strip()
        score = 0.0

        tag_match = _TAG_RE.match(text)
        if tag_match and tag_match.group(1).lower() in _VALID_TAGS:
            score += 0.25
            tag_str = tag_match.group(0)  # e.g. "[smile]"

            before_think = text.split("<think>")[0]
            opening = before_think[len(tag_str):].strip()
            if len(opening) > 5:
                score += 0.25

            if "<think>" in text and "</think>" in text:
                score += 0.25

                after_think = _THINK_RE.sub("", text).strip()
                full_resp = after_think[len(tag_str):].strip() if after_think.startswith(tag_str) else after_think
                if len(full_resp.split()) > 5:
                    score += 0.25

        rewards.append(score)
    return rewards


def length_reward_func(completions, **kwargs) -> list[float]:
    """
    Penalise outputs that are structurally impossible (too short) or
    uselessly verbose. Neutral in the middle — format reward does the
    heavy lifting.
    """
    rewards = []
    for completion in completions:
        words = len(completion.split())
        if words < 20:    # too short to contain tag + think + response
            rewards.append(-0.3)
        elif words > 400: # absurdly long
            rewards.append(-0.2)
        else:
            rewards.append(0.1)
    return rewards


def clinical_reward_func(completions, **kwargs) -> list[float]:
    """Penalise clinical jargon — Lumi should sound like a companion, not a clinician."""
    clinical_words = [
        "diagnosis", "symptom", "medication", "treatment",
        "patient", "disease", "prescribe", "cognitive", "dementia",
    ]
    rewards = []
    for completion in completions:
        if any(w in completion.lower() for w in clinical_words):
            rewards.append(-0.3)
        else:
            rewards.append(0.0)
    return rewards


# Select active reward functions based on training phase
if TRAINING_PHASE == 1:
    reward_funcs = [format_reward_func]
    output_dir   = "./lumi-grpo-phase1-format"
    print("Phase 1: format compliance reward ONLY")
else:
    reward_funcs = [format_reward_func, length_reward_func, clinical_reward_func]
    output_dir   = "./lumi-grpo-phase2-full"
    print("Phase 2: format + length + clinical rewards")

# =============================================================================
# Training Config
# =============================================================================

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_prompt_length=512,
    max_completion_length=512,   # must fit: tag + opening + <think>...</think> + full response
    num_train_epochs=1,
    fp16=True,
    logging_steps=5,
    save_steps=100,
    report_to="none",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

print(f"Initialising GRPOTrainer (phase={TRAINING_PHASE})...")
trainer = GRPOTrainer(
    model=BASE_MODEL,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

print("Starting GRPO training...")
trainer.train()

print(f"Saving model to {output_dir}/final")
trainer.save_model(f"{output_dir}/final")
print("Done.")

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

BASE_MODEL  = "Debdeep30/lumi-qwen3-4b"   # SFT model — Phase 1 barely updated weights
DATA_REPO   = "YUGOROU/lumi-data"
DATA_CONFIG = "filtered"
HF_TOKEN    = os.getenv("HF_TOKEN", "")

# TRAINING_PHASE controls which reward functions are active:
#   1 = format compliance ONLY — run this first until compliance > 95%
#   2 = all rewards — add after Phase 1 checkpoint is stable
TRAINING_PHASE = 2

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

_TAG_RE     = re.compile(r"\[(smile|nod|concerned|gentle|laugh)\]", re.IGNORECASE)
_THINK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL)
_VALID_TAGS = {"smile", "nod", "concerned", "gentle", "laugh"}


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Graded reward (0.0–1.0) for Qwen3's native format:
      <think>...</think>
      [avatar_tag] Opening line.
      Full response.

    +0.25  <think>...</think> block is present
    +0.25  valid avatar tag present anywhere in the completion
    +0.25  non-empty opening line after the tag
    +0.25  full response is at least 5 words after stripping think + tag

    Using graded partial credit gives the model a gradient signal even when
    it's only partially compliant — binary rewards stall early training.
    """
    rewards = []
    for completion in completions:
        text = completion.strip()
        score = 0.0

        has_think = "<think>" in text and "</think>" in text
        if has_think:
            score += 0.25

        tag_match = _TAG_RE.search(text)
        if tag_match and tag_match.group(1).lower() in _VALID_TAGS:
            score += 0.25
            tag_str = f"[{tag_match.group(1).lower()}]"

            clean = _THINK_RE.sub("", text).strip()
            body = clean.replace(tag_str, "").strip()
            lines = [l.strip() for l in body.split("\n") if l.strip()]

            if lines and len(lines[0]) > 5:
                score += 0.25   # opening line present

            if len(body.split()) > 5:
                score += 0.25   # full response present

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
    max_completion_length=512,   # must fit: <think>...</think> + [tag] + opening + full response
    num_generations=4,           # group size for GRPO advantage estimation
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

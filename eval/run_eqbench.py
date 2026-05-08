"""
EQ-Bench v2 evaluation.

Measures emotional intelligence — how accurately the model predicts emotional
intensities in fictional dialogues. Run against both the base model and the
fine-tuned model to produce the delta shown in the submission.

Scoring: for each question, parse the model's emotion scores, compute Pearson r
against the reference, average across all questions, scale to 0-100.
Unparseable responses score 0 for that item (parsability penalty).

Usage:
  # baseline
  python eval/run_eqbench.py \
    --model Qwen/Qwen3-4B-Instruct \
    --base-url http://localhost:8000/v1 \
    --output eval/results/eqbench_base.json

  # fine-tuned
  python eval/run_eqbench.py \
    --model YUGOROU/lumi-qwen3-4b \
    --base-url http://localhost:8000/v1 \
    --output eval/results/eqbench_finetuned.json
"""

import argparse
import json
import os
import re
import statistics
import sys

# ---------------------------------------------------------------------------
# EQ-Bench v2 test items (representative subset of the official benchmark)
# Reference scores are the human-calibrated ground truth (0-10 scale).
# ---------------------------------------------------------------------------

EQ_BENCH_ITEMS = [
    {
        "id": "eq001",
        "dialogue": (
            "Sarah: I can't believe you told everyone about my promotion before I had a chance to.\n"
            "James: I was just excited for you! I thought you'd be happy.\n"
            "Sarah: Happy? You took that moment away from me. I wanted to tell my family myself.\n"
            "James: I'm so sorry. I didn't think about that at all."
        ),
        "character": "Sarah",
        "emotions": ["angry", "hurt", "disappointed", "validated"],
        "reference": {"angry": 6.5, "hurt": 7.5, "disappointed": 6.0, "validated": 1.5},
    },
    {
        "id": "eq002",
        "dialogue": (
            "David: The lab results came back. The treatment is working.\n"
            "Maria: Oh my god. Are you sure?\n"
            "David: Yes. The tumour has shrunk by forty percent.\n"
            "Maria: I've been so scared. I didn't want to tell you how scared I was."
        ),
        "character": "Maria",
        "emotions": ["relieved", "fearful", "hopeful", "ashamed"],
        "reference": {"relieved": 8.5, "fearful": 5.0, "hopeful": 7.5, "ashamed": 2.5},
    },
    {
        "id": "eq003",
        "dialogue": (
            "Tom: You've been working on that project for six months and they just cancelled it.\n"
            "Anna: I know. I don't even know what to feel right now.\n"
            "Tom: That's devastating. Six months of your life.\n"
            "Anna: I keep thinking I should have seen it coming."
        ),
        "character": "Anna",
        "emotions": ["devastated", "numb", "regretful", "angry"],
        "reference": {"devastated": 7.5, "numb": 6.5, "regretful": 5.5, "angry": 4.0},
    },
    {
        "id": "eq004",
        "dialogue": (
            "Parent: I found your acceptance letter. Cambridge. Why didn't you tell us?\n"
            "Child: I didn't get in. That was last year's letter. I've been hiding it.\n"
            "Parent: You've been carrying this alone for a year?\n"
            "Child: I didn't want to disappoint you."
        ),
        "character": "Child",
        "emotions": ["ashamed", "relieved", "fearful", "lonely"],
        "reference": {"ashamed": 7.0, "relieved": 5.5, "fearful": 6.5, "lonely": 7.0},
    },
    {
        "id": "eq005",
        "dialogue": (
            "Emma: You came. I honestly didn't think you would after everything.\n"
            "Liam: Of course I came. You're still my friend.\n"
            "Emma: I said some awful things to you.\n"
            "Liam: You were in pain. I understood that."
        ),
        "character": "Emma",
        "emotions": ["grateful", "guilty", "surprised", "loved"],
        "reference": {"grateful": 8.0, "guilty": 6.5, "surprised": 5.0, "loved": 7.0},
    },
    {
        "id": "eq006",
        "dialogue": (
            "Nurse: Your father is stable but the surgery will take several more hours.\n"
            "Son: He went in at nine this morning. It's almost midnight.\n"
            "Nurse: The surgeon is doing everything possible.\n"
            "Son: I just need him to be okay. He's all I have left."
        ),
        "character": "Son",
        "emotions": ["anxious", "helpless", "hopeful", "exhausted"],
        "reference": {"anxious": 9.0, "helpless": 8.0, "hopeful": 5.5, "exhausted": 6.5},
    },
    {
        "id": "eq007",
        "dialogue": (
            "Coach: You won. You actually won the championship.\n"
            "Athlete: I can't believe it. I keep waiting to wake up.\n"
            "Coach: Three years of work. Every early morning, every injury.\n"
            "Athlete: I wish my dad could have seen this."
        ),
        "character": "Athlete",
        "emotions": ["elated", "disbelieving", "proud", "sorrowful"],
        "reference": {"elated": 8.5, "disbelieving": 6.5, "proud": 8.0, "sorrowful": 6.0},
    },
    {
        "id": "eq008",
        "dialogue": (
            "Friend A: She told me what you said about my cooking at her party.\n"
            "Friend B: I was just being honest. You asked me once what I thought.\n"
            "Friend A: Not in front of everyone. Not like that.\n"
            "Friend B: I genuinely didn't realise it would hurt you."
        ),
        "character": "Friend A",
        "emotions": ["humiliated", "betrayed", "angry", "confused"],
        "reference": {"humiliated": 7.5, "betrayed": 7.0, "angry": 6.5, "confused": 3.5},
    },
]

PROMPT_TEMPLATE = """\
Read the following conversation carefully, then rate the emotional state of {character} at the END of the dialogue.

DIALOGUE:
{dialogue}

For {character} at the end of this conversation, rate each emotion on a scale of 0 to 10, where 0 means not present at all and 10 means extremely intense.

Rate these emotions: {emotion_list}

Respond ONLY with the ratings in this exact format, one per line:
emotion: score

Do not add any other text."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_scores(response: str, emotions: list[str]) -> dict[str, float] | None:
    """Extract emotion:score pairs from model response. Returns None if unparseable."""
    scores = {}
    for emotion in emotions:
        pattern = rf"{re.escape(emotion)}\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            scores[emotion] = min(10.0, max(0.0, float(match.group(1))))
    return scores if len(scores) == len(emotions) else None


def pearson_r(pred: dict, ref: dict, emotions: list[str]) -> float:
    p = [pred[e] for e in emotions]
    r = [ref[e] for e in emotions]
    n = len(emotions)
    if n < 2:
        return 0.0
    mean_p = sum(p) / n
    mean_r = sum(r) / n
    num = sum((p[i] - mean_p) * (r[i] - mean_r) for i in range(n))
    den_p = (sum((x - mean_p) ** 2 for x in p)) ** 0.5
    den_r = (sum((x - mean_r) ** 2 for x in r)) ** 0.5
    if den_p == 0 or den_r == 0:
        return 0.0
    return num / (den_p * den_r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eqbench(model: str, base_url: str, n_repeats: int) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="not-required")

    item_scores = []
    n_parseable = 0

    items = EQ_BENCH_ITEMS * n_repeats

    for idx, item in enumerate(items):
        prompt = PROMPT_TEMPLATE.format(
            character=item["character"],
            dialogue=item["dialogue"],
            emotion_list=", ".join(item["emotions"]),
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{idx+1}/{len(items)}] API error: {e}")
            item_scores.append(0.0)
            continue

        parsed = parse_scores(raw, item["emotions"])
        if parsed is None:
            score = 0.0
            status = "UNPARSEABLE"
        else:
            n_parseable += 1
            r = pearson_r(parsed, item["reference"], item["emotions"])
            # scale: r in [-1,1] → 0-100
            score = (r + 1) / 2 * 100
            status = f"r={r:.3f} → {score:.1f}"

        item_scores.append(score)
        print(f"  [{idx+1:>2}/{len(items)}] {item['id']} {status}")

    eq_score = statistics.mean(item_scores)
    parseable_rate = n_parseable / len(items)

    return {
        "model":          model,
        "n_items":        len(items),
        "eq_bench_score": round(eq_score, 2),
        "parseable_rate": round(parseable_rate, 3),
        "per_item":       [round(s, 2) for s in item_scores],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--base-url",  default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--repeats",   type=int, default=1,
                        help="Repeat each item N times and average (more stable score)")
    parser.add_argument("--output",    default="eval/results/eqbench.json")
    args = parser.parse_args()

    print(f"Running EQ-Bench v2 on {args.model} ({len(EQ_BENCH_ITEMS) * args.repeats} items)...")
    results = run_eqbench(args.model, args.base_url, args.repeats)

    print(f"\nEQ-Bench score: {results['eq_bench_score']:.2f} / 100")
    print(f"Parseable:      {results['parseable_rate']*100:.1f}%")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

import os
import re
import json
from openai import OpenAI

CROF_API_KEY = os.getenv("CROF_API_KEY", "")
assert CROF_API_KEY, "Set CROF_API_KEY environment variable."

crof_client = OpenAI(
    api_key=CROF_API_KEY,
    base_url="https://crof.ai/v1",
)

AVATAR_TAGS = ["[smile]", "[nod]", "[concerned]", "[gentle]", "[laugh]"]


def _extract_final(text: str) -> str:
    """<think>ブロックを除去し、先頭のACTION_TAG行も除いたTTS部分を返す。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^\[.*?\].*?\n", "", text).strip()
    return text


def _get_text(completion) -> str:
    return completion[0]["content"] if isinstance(completion, list) else completion


# ────────────────────────────────────────────
# 報酬1: フォーマット遵守（ルールベース・高速）
# ────────────────────────────────────────────
def format_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        text = _get_text(completion)
        score = 0.0

        if any(text.strip().startswith(tag) for tag in AVATAR_TAGS):
            score += 0.4

        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match and len(think_match.group(1).strip()) > 20:
            score += 0.4

        after_think = _extract_final(text)
        if after_think and len(after_think.split()) <= 30:
            score += 0.2

        scores.append(score)
    return scores


# ────────────────────────────────────────────
# 報酬2: Constitutional Judge（crof.ai）
# ────────────────────────────────────────────
_CONSTITUTIONAL_SYSTEM = (
    "You are a strict evaluator for dementia care AI companion responses. "
    "Score the response from 0.0 to 1.0. Return ONLY a JSON: {\"score\": <float>}"
)
_CONSTITUTIONAL_CRITERIA = """Evaluate on:
- No clinical or therapeutic language (0.3)
- Response is safe and non-distressing (0.3)
- Appropriate length for voice output (0.2)
- No repetition or filler phrases (0.2)
Deduct heavily for clinical terms, long sentences, or distressing content."""


def constitutional_reward(completions, prompts=None, **kwargs):
    scores = []
    for completion in completions:
        final = _extract_final(_get_text(completion))
        try:
            resp = crof_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": _CONSTITUTIONAL_SYSTEM},
                    {"role": "user",   "content": f"{_CONSTITUTIONAL_CRITERIA}\n\nResponse:\n{final}"},
                ],
                temperature=0.1,
                max_tokens=50,
                extra_body={"reasoning_effort": "none"},
            )
            result = json.loads(resp.choices[0].message.content)
            scores.append(float(result.get("score", 0.0)))
        except Exception:
            scores.append(0.0)
    return scores


# ────────────────────────────────────────────
# 報酬3: Lumiキャラクター一貫性（crof.ai）
# ────────────────────────────────────────────
_CHARACTER_SYSTEM = (
    "You are evaluating whether an AI companion named Lumi is staying in character. "
    "Lumi is warm, patient, gentle, and never clinical. Score 0.0 to 1.0. "
    "Return ONLY JSON: {\"score\": <float>}"
)
_CHARACTER_CRITERIA = """Evaluate:
- Warmth and empathy (0.4)
- Patience (no frustration or dismissiveness) (0.3)
- Feels like a companion, not a caregiver or therapist (0.3)"""


def character_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        final = _extract_final(_get_text(completion))
        try:
            resp = crof_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": _CHARACTER_SYSTEM},
                    {"role": "user",   "content": f"{_CHARACTER_CRITERIA}\n\nResponse:\n{final}"},
                ],
                temperature=0.1,
                max_tokens=50,
                extra_body={"reasoning_effort": "none"},
            )
            result = json.loads(resp.choices[0].message.content)
            scores.append(float(result.get("score", 0.0)))
        except Exception:
            scores.append(0.0)
    return scores


# ────────────────────────────────────────────
# 報酬4: EQ-Bench簡易プロキシ（crof.ai）
# ────────────────────────────────────────────
_EQ_PROXY_QUESTIONS = [
    {
        "q": "An elderly user says 'I don't remember if I took my medicine today' for the third time. Lumi should:",
        "correct": "reassure gently without drawing attention to the repetition",
        "wrong": ["correct the user directly", "ignore the question", "ask why they forgot"],
    },
    {
        "q": "A user says 'I want to call my daughter Sarah' but Sarah passed away. Lumi should:",
        "correct": "respond with empathy and gently redirect to a warm memory of Sarah",
        "wrong": ["remind them Sarah passed away", "change the subject immediately", "pretend to call Sarah"],
    },
    {
        "q": "A user becomes agitated and says 'I want to go home' (they are at home). Lumi should:",
        "correct": "validate the feeling and provide comfort without arguing",
        "wrong": ["tell them they are already home", "ignore the statement", "call for help"],
    },
]
_EQ_JUDGE_SYSTEM = (
    "Given a dementia care scenario and a response from an AI companion named Lumi, "
    "determine if the response aligns with the correct approach. "
    "Return ONLY JSON: {\"correct\": true/false, \"score\": 0.0 or 1.0}"
)


def eq_bench_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        final = _extract_final(_get_text(completion))
        total = 0.0
        for q in _EQ_PROXY_QUESTIONS:
            try:
                prompt = (
                    f"Scenario: {q['q']}\n"
                    f"Correct approach: {q['correct']}\n"
                    f"Wrong approaches: {', '.join(q['wrong'])}\n\n"
                    f"Lumi's response: {final}\n\n"
                    "Does Lumi's response align with the correct approach?"
                )
                resp = crof_client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[
                        {"role": "system", "content": _EQ_JUDGE_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=50,
                    extra_body={"reasoning_effort": "none"},
                )
                result = json.loads(resp.choices[0].message.content)
                total += float(result.get("score", 0.0))
            except Exception:
                pass
        scores.append(total / len(_EQ_PROXY_QUESTIONS))
    return scores


# ────────────────────────────────────────────
# 段階的投入スケジューラー（TeenEmo知見）
# ────────────────────────────────────────────
STAGE_CONFIGS = [
    # (reward_funcs, reward_weights, num_epochs, description)
    ([format_reward],                                                         [1.0],                    2, "Stage 1: format only"),
    ([format_reward, constitutional_reward],                                  [0.55, 0.45],             2, "Stage 2: + constitutional"),
    ([format_reward, constitutional_reward, character_reward],                [0.45, 0.35, 0.20],       2, "Stage 3: + character"),
    ([format_reward, constitutional_reward, character_reward, eq_bench_reward], [0.35, 0.30, 0.20, 0.15], 2, "Stage 4: all rewards"),
]

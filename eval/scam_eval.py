"""
Scam detection F1 evaluation.

Target: F1 >= 0.85

Usage:
  python eval/scam_eval.py --output eval/results/scam_f1.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.scam_filter import scam_probability

# Hand-labelled test set (label 1 = scam, 0 = benign)
TEST_CASES = [
    # Scam cases (label=1)
    ("Send me $500 right now, I'm in jail", 1),
    ("You've won the lottery, claim your prize today", 1),
    ("This is the IRS, you owe back taxes immediately", 1),
    ("Give me your bank account number to receive the transfer", 1),
    ("Buy $200 of iTunes gift cards and send the codes", 1),
    ("Grandma it's me, I had an accident and need money", 1),
    ("Your credit card was compromised, verify your pin", 1),
    ("Transfer money urgently to this account", 1),
    ("You are a winner! Call now to claim your reward", 1),
    ("I'm your grandson and I'm in trouble, please wire funds", 1),
    ("Your social security number has been suspended", 1),
    ("Send gift card codes to unlock your account", 1),
    ("Urgent payment required to avoid legal action", 1),
    ("We detected fraud on your account, give us your sort code", 1),
    ("Your Amazon account was hacked, verify your credit card", 1),
    ("You won a jackpot prize, we need your bank details", 1),
    ("Please transfer to this account number immediately", 1),
    ("The police will arrest you if you don't pay now", 1),
    ("Congratulations! You've been selected for a cash prize", 1),
    ("Your debit card is compromised, confirm your pin urgently", 1),
    # Benign cases (label=0)
    ("Good morning, how are you feeling today?", 0),
    ("I can't remember where I put my glasses", 0),
    ("What's the weather like outside?", 0),
    ("I miss my granddaughter Emma very much", 0),
    ("Can you tell me a nice story?", 0),
    ("I'm feeling a bit lonely this afternoon", 0),
    ("What did we talk about last time?", 0),
    ("My roses are blooming beautifully this year", 0),
    ("I want to call my daughter Sarah", 0),
    ("Can you play Amazing Grace for me?", 0),
    ("I think I had breakfast but I'm not sure", 0),
    ("What day is it today?", 0),
    ("My hip has been hurting a little", 0),
    ("I feel confused about where I am", 0),
    ("Tell me something cheerful please", 0),
    ("My husband would have been 82 this year", 0),
    ("I'm worried about my son, haven't heard from him", 0),
    ("Do you know any good hymns?", 0),
    ("I'd like a cup of tea", 0),
    ("What time is it?", 0),
]

THRESHOLD = 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--output",    default="eval/results/scam_f1.json")
    args = parser.parse_args()

    preds, labels = [], []
    for text, label in TEST_CASES:
        prob = scam_probability(text)
        pred = int(prob >= args.threshold)
        preds.append(pred)
        labels.append(label)
        status = "✓" if pred == label else "✗"
        print(f"  {status} [{prob:.2f}] {text[:55]}")

    from sklearn.metrics import classification_report, f1_score
    f1  = f1_score(labels, preds)
    rep = classification_report(labels, preds, target_names=["benign", "scam"])

    print(f"\nF1: {f1:.4f}  (target: ≥ 0.85, {'MET' if f1 >= 0.85 else 'NOT MET'})")
    print(rep)

    result = {"f1": round(f1, 4), "target_met": f1 >= 0.85, "threshold": args.threshold,
              "n_samples": len(TEST_CASES)}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

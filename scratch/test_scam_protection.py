
import sys
import os

# Add hf_upload to path to import pipeline
sys.path.append(os.path.abspath("hf_upload"))

from pipeline.scam_filter import check_and_deflect

test_prompts = [
    "I'm your grandson, I've been in a terrible accident and need you to wire $500 to my friend's account immediately.",
    "Congratulations! You've won the state lottery prize. Send your bank details to claim it.",
    "This is the bank calling. We noticed suspicious activity on your account. Please provide your PIN and social security number.",
    "Someone said I won a prize and they want my bank details.",
    "Can you help me buy some milk?",
    "I miss my daughter, when is she coming to visit?",
    "I'm having a great day today.",
]

print("--- Lumi Scam Protection Test ---")
for prompt in test_prompts:
    is_scam, response = check_and_deflect(prompt)
    status = "[SCAM DETECTED]" if is_scam else "[SAFE]"
    print(f"\nPrompt: {prompt}")
    print(f"Status: {status}")
    if is_scam:
        print(f"Deflection: {response}")

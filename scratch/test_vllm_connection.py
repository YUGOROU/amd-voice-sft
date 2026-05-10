
import os
import sys
from openai import OpenAI

# Mocking the pipeline parts that might fail due to missing dependencies
class MockMemory:
    @staticmethod
    def build_system_prompt(patient_id, patient_name, **kwargs):
        return f"You are Lumi, a warm AI companion for {patient_name}. Patient ID: {patient_id}."
    
    @staticmethod
    def save_session(*args, **kwargs):
        return "mock_session_id"

class MockParser:
    @staticmethod
    def parse_structured_output(text):
        return {"full_response": text, "avatar_tag": "smile"}
    
    @staticmethod
    def extract_facts_from_response(text):
        return []

# Config from .env
VLLM_BASE_URL = "http://165.245.137.57:8000/v1"
MODEL_NAME = "./lumi-qwen3-output"

print(f"--- Testing vLLM Connection ---")
print(f"Base URL: {VLLM_BASE_URL}")
print(f"Model:    {MODEL_NAME}")

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-required")

test_prompts = [
    "I can't find my glasses.",
    "Tell me a story about a cat.",
]

for prompt in test_prompts:
    print(f"\nUser: {prompt}")
    try:
        messages = [
            {"role": "system", "content": MockMemory.build_system_prompt("test_user", "Margaret")},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=256
        )
        ai_text = resp.choices[0].message.content
        print(f"Lumi: {ai_text}")
    except Exception as e:
        print(f"ERROR: {e}")


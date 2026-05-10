
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Use the same config as the app
VLLM_BASE_URL = "http://134.199.192.73:8000/v1"
MODEL_NAME = "Debdeep30/lumi-qwen2.5-3b"
API_KEY = "LumiAI"

client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)

print(f"Testing Lumi at {VLLM_BASE_URL}...")

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are Lumi, a warm and patient AI companion for elderly people."},
            {"role": "user", "content": "Hi Lumi, I can't find my glasses."}
        ],
        temperature=0.7,
        max_tokens=100,
        extra_body={"repetition_penalty": 1.2}
    )
    print("\nLumi's Response:")
    print("-" * 30)
    print(response.choices[0].message.content)
    print("-" * 30)
except Exception as e:
    print(f"Error connecting to Lumi: {e}")

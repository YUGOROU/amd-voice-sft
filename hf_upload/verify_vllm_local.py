
import requests
import json

def check_vllm():
    url = "http://localhost:8000/v1/chat/completions"
    model = "./lumi-qwen3-output"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Lumi."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    print(f"Checking vLLM at {url}...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ SUCCESS! vLLM responded correctly.")
            print(f"Response: {response.json()['choices'][0]['message']['content']}")
        else:
            print(f"❌ FAILED! Status Code: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")

if __name__ == "__main__":
    check_vllm()

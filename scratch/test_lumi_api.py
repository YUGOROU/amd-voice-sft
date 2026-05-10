import asyncio
from gradio_client import Client

async def test_lumi():
    print("🚀 Connecting to Lumi on HF Spaces...")
    try:
        # Connect specifically to the mounted Gradio instance
        client = Client("https://debdeep30-lumi-voice-companion.hf.space/gradio/")
        
        test_prompts = [
            "Hi Lumi, who are you?",
            "I'm feeling a bit confused today, where am I?",
            "Can you remind me what I should do this afternoon?"
        ]
        
        history = []
        profile_id = "lumi"
        
        for prompt in test_prompts:
            print(f"\n👤 User: {prompt}")
            # lumi_api(message, history, profile_id)
            result = client.predict(
                prompt,
                history,
                profile_id,
                api_name="/lumi_api"
            )
            
            response_text, audio_path, avatar_tag = result
            print(f"☀️ Lumi: {response_text}")
            print(f"🎭 Mood: {avatar_tag}")
            print(f"🔊 Audio: {audio_path}")
            
            # Update history for next turn
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response_text})
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_lumi())

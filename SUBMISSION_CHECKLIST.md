# Lumi — Remaining Tasks to HF Space Submission
**AMD Developer Hackathon 2026 · Fine-Tuning Track**
_Last updated: 2026-05-09 08:30 IST_

---

## Current State Summary
| Service | Status | Location |
|---|---|---|
| vLLM (LLM inference) | ✅ Running | droplet:8000 |
| STT (faster-whisper) | ✅ Running | droplet:8001 |
| TTS (edge-tts) | ✅ Running + systemd | droplet:8002 |
| SadTalker (talking head) | ✅ Running + systemd | droplet:8003 |
| HF Space | ✅ **LIVE & WORKING** | Debdeep30/lumi-voice-companion |
| Model | ✅ Pushed | Debdeep30/lumi-qwen3-4b-grpo |
| Portraits | ✅ All 10 uploaded to Space | portraits/*.png |
| SadTalker video | ✅ **583 KB MP4 generated** | Tested 2026-05-09 |
| Branch | ✅ feature/lumi-core up to date | All pushed |

---

## ✅ COMPLETED — Critical Items

### 1. ~~Merge branches~~ ✅
All pipeline files (profiles.py, talking_head.py, all portraits) are already in the Space.
HF Space is RUNNING at https://debdeep30-lumi-voice-companion.hf.space/

### 2. ~~End-to-end SadTalker test~~ ✅ FIXED 2026-05-09
Fixed 3 numpy 2.x compatibility issues in SadTalker:
- `np.VisibleDeprecationWarning` → commented out (preprocess.py:12)
- `--enhancer gfpgan` removed from sadtalker_server.py args (GFPGAN not installed)
- `np.float` → `np.float64` in my_awing_arch.py:18
- `np.array([w0,h0,s,t[0],t[1]])` → float() wrap in preprocess.py:101
**Result: 583 KB valid MP4 generated ✅**

### 3. ~~Verify gr.Video in Space~~ — NEXT STEP
Test the live voice tab in the Space:
- Speak a sentence in the 🎤 Live Voice Chat tab
- Confirm talking head video appears (TALKING_HEAD_URL secret is set)
- Confirm static portrait returns after video ends

### 4. ChromaDB persistence — ACCEPTABLE FOR DEMO
Using Option A (in-session memory only). ChromaDB resets on Space restart but that's fine for the hackathon demo.

---

## 🟡 IMPORTANT — Fix Before Recording Demo Video

### 5. Portrait quality check
Open each portrait in `hf_upload/portraits/` and verify:
- [ ] Front-facing face (SadTalker needs clear frontal view)
- [ ] No distortions / extra limbs (SD artifacts)
- [ ] Face is in the centre of the frame
- [ ] Good contrast against background

If a portrait is bad, regenerate it:
```bash
ssh root@165.245.129.208
# Edit /root/gen_portraits.py, remove the "skip if exists" check for that ID
# Then: /root/sadtalker_env/bin/python /root/gen_portraits.py
# scp the new file back
```

### 6. SadTalker preprocess mode
The current server uses `--preprocess full` which keeps the whole portrait.
For cropped-to-face results (often sharper lip sync), try `--preprocess crop`.
Test both and use whichever looks better.

### 7. Loading state during SadTalker generation
SadTalker takes 10-15s. The UI currently just waits silently.
Add a status message so the user knows something is happening:

In `live_stream()`, after STT returns user_text, yield an interim status:
```python
# Quick yield to show "Lumi is thinking..."
# Then do LLM + TTS + SadTalker
```
This requires converting `live_stream` to a generator — medium complexity.
**For hackathon, acceptable to leave as-is and note the latency.**

### 8. Family Dashboard — include voice chat history
Currently `Generate Summary` only reads `chatbot1` (text tab).
The voice chat has its own `chatbot2`. Wire both:
```python
summary_btn.click(end_session, [chatbot1, chatbot2], [summary_out])
# Update end_session() to accept two history lists and merge them
```

### 9. ~~Droplet services — auto-restart~~ ✅ DONE
Systemd services created and enabled:
- `/etc/systemd/system/lumi-tts.service` → port 8002
- `/etc/systemd/system/lumi-sadtalker.service` → port 8003
Both set to `Restart=always`.

---

## 🟢 NICE TO HAVE — Polish Before Submission

### 10. Per-profile voice test
Use the "🎭 Choose Companion" tab to switch from Sophie to Dorothy.
Send the same prompt to both and verify:
- Dorothy's voice is noticeably slower and lower pitched
- Marcus's voice is different from James's

### 11. Scam detection test in voice chat
Say "Someone called and said I won a prize, they want my bank details."
Verify:
- Lumi deflects without alarming the patient
- The gentle avatar is shown

### 12. Model card update
On `Debdeep30/lumi-qwen3-4b-grpo` HF page, update the model card to mention:
- SadTalker talking head integration
- 10 diverse avatar profiles
- EQ-Bench 91.22/100 result
- AMD MI300X end-to-end pipeline

### 13. Space README update
Update `hf_upload/README.md`:
- Add `TALKING_HEAD_URL` to the secrets table
- Mention the 10 profiles
- Update feature list to include talking head

### 14. About tab in the app
Update the About tab text to reflect the final architecture including:
- SadTalker on AMD MI300X
- 10 diverse profiles
- SD-generated portraits

---

## 🎬 Demo Video — 4 Required Scenes

Record a 3–5 minute video showing:

| # | Scene | What to show |
|---|---|---|
| 1 | **Problem** | Elderly isolation + dementia challenges. 30 seconds. Can be slides. |
| 2 | **Memory continuity** | Start a text chat. Mention your name and a detail ("I love roses"). End session. Start a new one. Show Lumi remembering. |
| 3 | **Scam deflection** | In voice chat: say the bank details scam prompt. Show Lumi deflecting calmly. Show the "gentle" avatar. |
| 4 | **Talking head + live voice** | Switch to Sophie or Marcus profile. Start live voice chat. Speak. Show the animated talking head video play back with Lumi's voice. |
| 5 | **Tech architecture** (optional) | Brief screen share: MI300X rocm-smi, vLLM endpoint, SadTalker server, EQ-Bench score. |

**Tips:**
- Pre-warm the Space before recording (open it in browser, wait 60s)
- For Scene 4, speak slowly and pause clearly so VAD triggers correctly
- SadTalker output: have a pre-generated example ready as backup in case of latency

---

## 📋 HF Space Secrets Checklist

Verify all are set at `https://huggingface.co/spaces/Debdeep30/lumi-voice-companion/settings`:

| Secret | Value | Status |
|---|---|---|
| `VLLM_BASE_URL` | `http://165.245.129.208:8000/v1` | ✅ Set |
| `MODEL_NAME` | `Debdeep30/lumi-qwen3-4b-grpo` | ✅ Set |
| `PATIENT_NAME` | `Margaret` (or patient's name) | ✅ Set |
| `STT_URL` | `http://165.245.129.208:8001` | ✅ Set |
| `TTS_URL` | `http://165.245.129.208:8002` | ✅ Set |
| `TALKING_HEAD_URL` | `http://165.245.129.208:8003` | ✅ Set |
| `PATIENT_ID` | `demo_user_001` (optional, has default) | — |

---

## 📦 Hackathon Submission Checklist

- [ ] HF Space is live and publicly accessible
- [ ] Demo video uploaded (YouTube or direct link)
- [ ] GitHub repo URL: `https://github.com/YUGOROU/amd-voice-sft`
- [ ] HF Model URL: `https://huggingface.co/Debdeep30/lumi-qwen3-4b-grpo`
- [ ] HF Space URL: `https://huggingface.co/spaces/Debdeep30/lumi-voice-companion`
- [ ] Submit at hackathon portal with all above links
- [ ] EQ-Bench score mentioned (91.22/100)
- [ ] AMD MI300X mentioned as training + inference hardware

---

## Priority Order (remaining)

1. ~~**Merge branches**~~ ✅
2. ~~**SadTalker end-to-end test**~~ ✅ (583 KB MP4 confirmed)
3. **Portrait quality check** — open portraits in browser, verify faces are clear
4. ~~**Systemd services**~~ ✅
5. **Test live voice + talking head in the Space browser**
6. **Demo video recording** (4 scenes)
7. **Hackathon submission**

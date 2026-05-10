"""
Structured output parser for Lumi.

Qwen3 outputs in its native thinking-first format:
    <think>
    Internal reasoning — NEVER sent to TTS or shown to user.
    </think>
    [avatar_tag] Short opening line.
    Full warm companion response continues here.

TTS latency trick: stream until </think> appears, then fire TTS immediately
on the opening line that follows.

parse_structured_output() returns a dict with:
  - avatar_tag    : one of the 5 states (defaults to "smile" if missing)
  - opening_line  : fires TTS after </think> is seen (latency-hiding trick)
  - full_response : complete response text (think block stripped)
"""

import re

VALID_TAGS = {"smile", "nod", "concerned", "gentle", "laugh"}
_TAG_RE    = re.compile(r"\[(\w+)\]")           # anywhere in text
_TAG_START = re.compile(r"^\[(\w+)\]")          # at start of text
_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL)


_ACTION_RE = re.compile(r"\[\[ACTION:\s*([^\]]+)\]\]")


def parse_structured_output(text: str) -> dict:
    text = text.strip()

    # Strip think block first — works for both tag-first and think-first formats
    clean = _THINK_RE.sub("", text).strip()

    # --- action: extract [[ACTION: TYPE | ...]] tags ---
    action = None
    action_match = _ACTION_RE.search(clean)
    if action_match:
        raw_content = action_match.group(1).strip()
        parts = [p.strip() for p in raw_content.split("|")]
        if parts:
            action = {"type": parts[0], "payload": parts[1:]}
        # Clean the body for the user (remove the action tag)
        clean_body = _ACTION_RE.sub("", clean).strip()
    else:
        clean_body = clean

    # --- avatar tag: prefer at start of clean text, fall back to anywhere ----
    tag_match = _TAG_START.match(clean_body) or _TAG_RE.search(clean_body)
    if tag_match and tag_match.group(1).lower() in VALID_TAGS:
        avatar_tag = tag_match.group(1).lower()
    else:
        avatar_tag = "smile"

    # --- opening line: first line of the clean text after removing the tag ---
    tag_bracket = f"[{avatar_tag}]"
    body = clean_body.replace(tag_bracket, "").replace(tag_bracket.upper(), "").strip()
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    opening_line = lines[0] if lines else body[:80]
    full_response = body

    return {
        "avatar_tag": avatar_tag,
        "opening_line": opening_line,   # fire TTS after </think> token
        "full_response": full_response, # append to TTS queue
        "action": action
    }


def extract_facts_from_response(text: str) -> list[str]:
    """
    Heuristically extract memorable facts from a model response.
    Used to build the patient's persistent memory in ChromaDB.
    """
    facts = []
    lines = text.split(".")
    for line in lines:
        line = line.strip()
        # keep lines that mention personal details, preferences, family
        keywords = ("granddaughter", "grandson", "daughter", "son", "wife",
                    "husband", "likes", "loves", "favourite", "favorite",
                    "anniversary", "birthday", "garden", "music", "hymn",
                    "lonely", "feels", "misses", "remembers")
        if any(k in line.lower() for k in keywords) and len(line) > 15:
            facts.append(line)
    return facts[:5]  # cap at 5 per turn

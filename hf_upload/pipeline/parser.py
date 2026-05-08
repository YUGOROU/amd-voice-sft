"""
Structured output parser for Lumi.

Every model response follows the format trained by the crof pipeline:
    [avatar_tag] Short opening line.
    <think>
    Internal reasoning — NEVER sent to TTS or shown to user.
    </think>
    Full warm companion response continues here.

parse_structured_output() returns a dict with:
  - avatar_tag    : one of the 5 states (defaults to "smile" if missing)
  - opening_line  : fires TTS immediately (latency-hiding trick)
  - full_response : complete response text appended to TTS queue
"""

import re

VALID_TAGS = {"smile", "nod", "concerned", "gentle", "laugh"}
_TAG_RE    = re.compile(r"^\[(\w+)\]")
_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_structured_output(text: str) -> dict:
    text = text.strip()

    # --- avatar tag ----------------------------------------------------------
    tag_match = _TAG_RE.match(text)
    if tag_match and tag_match.group(1) in VALID_TAGS:
        avatar_tag = tag_match.group(1)
    else:
        avatar_tag = "smile"

    # --- opening line (everything before <think>, minus the tag) ------------
    before_think = text.split("<think>")[0]
    opening_line = before_think.replace(f"[{avatar_tag}]", "").strip()

    # --- strip think block entirely -----------------------------------------
    clean = _THINK_RE.sub("", text)
    full_response = clean.replace(f"[{avatar_tag}]", "").strip()

    return {
        "avatar_tag": avatar_tag,
        "opening_line": opening_line,   # fire TTS on this IMMEDIATELY
        "full_response": full_response, # append to TTS queue
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

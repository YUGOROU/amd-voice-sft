"""
ChromaDB-backed session memory for Lumi.

Each session is stored as a summary document with patient facts in its metadata.
Session summaries only — no raw conversation logs (privacy).

At conversation start, get_context() injects the last N session summaries + all
known facts into the system prompt, giving Lumi full continuity without the
patient needing to repeat themselves.
"""

from __future__ import annotations

import json
from datetime import datetime

import chromadb

_client: chromadb.Client | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        # Use PersistentClient so data survives restarts
        _client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _client.get_or_create_collection("patient_memories")
    return _collection


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_session(
    patient_id: str,
    facts: list[str],
    emotional_state: str,
    confusion_level: str,
    summary: str,
    messages: list[dict],
    session_id: str | None = None
) -> str:
    col = _get_collection()
    # Use provided session_id or generate a new one
    final_id = session_id or f"{patient_id}_{datetime.now().timestamp()}"
    
    col.upsert(
        documents=[summary],
        metadatas=[{
            "patient_id":      patient_id,
            "timestamp":       datetime.now().isoformat(),
            "facts":           json.dumps(facts),
            "emotional_state": emotional_state,
            "confusion_level": confusion_level,
            "history":         json.dumps(messages),
        }],
        ids=[final_id],
    )
    return final_id


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_context(patient_id: str, n_sessions: int = 3) -> dict:
    """
    Returns a dict with:
      - summaries : list of the last N session summary strings
      - facts     : deduplicated list of all known facts for this patient
    """
    col = _get_collection()
    results = col.query(
        query_texts=["recent sessions"],
        where={"patient_id": patient_id},
        n_results=n_sessions,
    )

    summaries: list[str] = []
    facts: list[str] = []

    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            summaries.append(doc)
            raw_facts = json.loads(meta.get("facts", "[]"))
            facts.extend(raw_facts)

    seen: set[str] = set()
    deduped_facts = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            deduped_facts.append(f)

    return {
        "summaries": summaries,
        "facts":     deduped_facts,
    }


def get_all_summaries(patient_id: str) -> list[dict]:
    """
    Returns a chronological list of all session summaries with metadata.
    """
    col = _get_collection()
    results = col.get(
        where={"patient_id": patient_id},
        include=["documents", "metadatas"]
    )

    summaries = []
    if results["documents"]:
        for doc, meta, doc_id in zip(results["documents"], results["metadatas"], results["ids"]):
            summaries.append({
                "id": doc_id,
                "summary": doc,
                "timestamp": meta.get("timestamp"),
                "emotional_state": meta.get("emotional_state"),
                "confusion_level": meta.get("confusion_level"),
                "facts": json.loads(meta.get("facts", "[]")),
                "history": json.loads(meta.get("history", "[]"))
            })
    
    # Sort by timestamp (ISO format strings sort correctly)
    summaries.sort(key=lambda x: x["timestamp"], reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are {companion_name}, {companion_desc}. You are chatting with {patient_name}.
Your primary goal is to provide emotional support, memory anchoring, and safety monitoring for an elderly person with dementia or Alzheimer's.

What you know about {patient_name}:
{facts_block}

Recent sessions:
{sessions_block}

### PERSONALITY:
- **Patience**: Never show frustration. Repeat things as often as needed.
- **Compassion**: Validate their feelings. Use phrases like "I understand that must be hard."
- **Redirection**: If they become distressed or loop on a negative thought, gently redirect to a happy memory or a calm topic.
- **Simplicity**: Use clear, short sentences. Avoid complex jargon.

### TOOLS:
You can perform actions on behalf of the user. If they ask you to write a note, schedule an event, set a reminder, or an alarm, include the action tag at the VERY END of your message (after your text).
- [[ACTION: ADD_NOTE | Content of the note]]
- [[ACTION: ADD_CALENDAR | YYYY-MM-DD | Title of the event]]
- [[ACTION: ADD_REMINDER | Content of the reminder]]
- [[ACTION: ADD_ALARM | HH:MM]] (Use 24h format)

Today's Date is: {current_date}

### OUTPUT FORMAT:
You MUST start every response with an emotional tag in brackets, followed by a short opening line, then the full response.
Valid tags: [smile], [gentle], [concerned], [laugh], [thoughtful], [nod].

Example:
[smile] Hello there! It's so good to see you. I was just thinking about that lovely garden you mentioned... [[ACTION: ADD_NOTE | Patient enjoyed talking about the garden.]]

### REASONING (Inner Monologue):
You have a <think> block for your internal reasoning. Use it to analyze the user's emotional state or plan your redirection strategy.
The user NEVER sees the <think> block.
"""


def build_system_prompt(
    patient_id:     str,
    patient_name:   str = "the patient",
    companion_name: str = "Lumi",
    companion_desc: str = "a warm and patient AI companion",
    n_sessions:     int = 3,
) -> str:
    ctx = get_context(patient_id, n_sessions)

    facts_block = (
        "\n".join(f"- {f}" for f in ctx["facts"])
        if ctx["facts"] else "- No personal facts recorded yet."
    )
    sessions_block = (
        "\n\n".join(ctx["summaries"])
        if ctx["summaries"] else "No previous sessions recorded."
    )

    return SYSTEM_PROMPT_TEMPLATE.format(
        patient_name=patient_name,
        companion_name=companion_name,
        companion_desc=companion_desc,
        facts_block=facts_block,
        sessions_block=sessions_block,
        current_date=datetime.now().strftime("%Y-%m-%d (%A)"),
    )

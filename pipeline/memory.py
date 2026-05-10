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
        _client = chromadb.Client()
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
) -> None:
    col = _get_collection()
    doc_id = f"{patient_id}_{datetime.now().timestamp()}"
    col.add(
        documents=[summary],
        metadatas=[{
            "patient_id":      patient_id,
            "timestamp":       datetime.now().isoformat(),
            "facts":           json.dumps(facts),
            "emotional_state": emotional_state,
            "confusion_level": confusion_level,
        }],
        ids=[doc_id],
    )


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


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are Lumi, a warm and patient companion for {patient_name}.

What you know about {patient_name}:
{facts_block}

Recent sessions:
{sessions_block}

Always respond in this exact format:
[avatar_tag] Short opening line.
<think>Your internal reasoning here. Never sent to user.</think>
Full warm response.

Be gentle and patient. Never correct temporal confusion — focus on the emotion, not the timeline.
"""


def build_system_prompt(
    patient_id: str,
    patient_name: str = "the patient",
    n_sessions: int = 3,
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
        facts_block=facts_block,
        sessions_block=sessions_block,
    )

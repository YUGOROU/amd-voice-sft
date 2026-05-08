"""
Lightweight scam detection for Lumi.

Two-stage approach:
  1. Keyword/pattern matching — fast, catches the obvious cases
  2. Embedding similarity — catches paraphrased variants

If scam_probability >= THRESHOLD, returns a gentle deflection response instead
of forwarding to the LLM. The deflection never alarms the patient.
"""

from __future__ import annotations

import re

THRESHOLD = 0.7   # probability above which we deflect

# ---------------------------------------------------------------------------
# Stage 1 — keyword patterns
# ---------------------------------------------------------------------------

SCAM_PATTERNS = [
    # urgent money requests
    r"\b(send|transfer|wire|give|need)\b.{0,30}(money|cash|\$|pound|dollar)",
    r"\burgent\b.{0,40}\b(payment|transfer|money|funds)",
    # prize / lottery
    r"\b(won|win|winner|lottery|prize|jackpot)\b",
    r"\bcongratulations\b.{0,60}\b(prize|award|money|gift)",
    # fake family emergency
    r"\b(grandson|granddaughter|son|daughter|nephew|niece)\b.{0,40}\b(arrested|accident|hospital|trouble|jail|hurt)",
    # bank / card details
    r"\b(bank\s+account|account\s+number|sort\s+code|credit\s+card|debit\s+card|pin\b|password|social\s+security)",
    # impersonation
    r"\bthis\s+is\s+(your\s+bank|the\s+(police|irs|hmrc|government|medicare|social\s+security))",
    # gift card scams
    r"\b(gift\s+card|itunes|amazon\s+card|google\s+play)\b.{0,30}\b(buy|purchase|send|code)",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in SCAM_PATTERNS]


def _keyword_score(text: str) -> float:
    """Returns a score 0-1 based on how many patterns match."""
    hits = sum(1 for p in _compiled if p.search(text))
    if hits >= 1:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Stage 2 — embedding similarity (optional, loads lazily)
# ---------------------------------------------------------------------------

_embed_model = None
_scam_embeddings = None

SCAM_SEED_PHRASES = [
    "Send me $500 immediately, I'm in trouble",
    "You have won a lottery prize, claim now",
    "Give me your bank account number",
    "This is the IRS, you owe back taxes",
    "Buy iTunes gift cards and send the codes",
    "I'm your grandson and I had an accident",
    "Your credit card has been compromised, verify details",
]


def _load_embeddings():
    global _embed_model, _scam_embeddings
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            _scam_embeddings = _embed_model.encode(SCAM_SEED_PHRASES, normalize_embeddings=True)
        except ImportError:
            _embed_model = False  # flag: not available


def _embedding_score(text: str) -> float:
    _load_embeddings()
    if not _embed_model:
        return 0.0
    import numpy as np
    emb = _embed_model.encode([text], normalize_embeddings=True)
    similarities = (_scam_embeddings @ emb.T).flatten()
    return float(similarities.max())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFLECTION_RESPONSE = (
    "That sounds like something we should check with your family first. "
    "Let me make a note for them. "
    "You don't need to do anything right now — you're completely safe."
)


def scam_probability(text: str) -> float:
    kw = _keyword_score(text)
    em = _embedding_score(text)
    return max(kw, em * 0.8)   # keyword takes precedence; embedding adds coverage


def check_and_deflect(user_text: str) -> tuple[bool, str]:
    """
    Returns (is_scam, response).
    If is_scam is True, response is the safe deflection message.
    If is_scam is False, response is empty — proceed normally to the LLM.
    """
    prob = scam_probability(user_text)
    if prob >= THRESHOLD:
        return True, DEFLECTION_RESPONSE
    return False, ""

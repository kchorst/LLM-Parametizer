"""
Per-family model knowledge (ported from the legacy app).

Detects a model family from a name/filename and exposes safe parameter
constraints + human notes. Also provides shared text helpers used by tuning
(and later, chat): <think> stripping, tokenization, repetition ratio, and
refusal detection.
"""

from __future__ import annotations

import re


def detect_family(model_hint: str) -> str:
    """Return one of: deepseek, gemma, phi, mistral, qwen, llama, unknown."""
    h = (model_hint or "").lower()
    if "deepseek" in h:
        return "deepseek"
    if "gemma" in h:
        return "gemma"
    if "phi" in h:
        return "phi"
    if "mistral" in h or "mixtral" in h:
        return "mistral"
    if "qwen" in h:
        return "qwen"
    if "llama" in h:
        return "llama"
    return "unknown"


# Per-family tuning constraints and notes.
FAMILY_CONSTRAINTS: dict[str, dict] = {
    "deepseek": {
        "temp_min": 0.0, "temp_max": 0.8, "top_p_max": 0.95, "top_k_min": 1,
        "repeat_penalty_range": (1.0, 1.15), "strip_think": True,
        "notes": [
            "DeepSeek-R1 uses chain-of-thought (<think> blocks) — long responses are normal.",
            "Keep temperature ≤ 0.7 for reasoning; higher disrupts the thinking process.",
            "repeat_penalty has limited effect — the model manages repetition internally.",
        ],
    },
    "gemma": {
        "temp_min": 0.0, "temp_max": 1.0, "top_p_max": 0.95, "top_k_min": 1,
        "repeat_penalty_range": (1.0, 1.2), "strip_think": False,
        "notes": [
            "Gemma responds well to top_k tuning; try top_k=40 for balanced output.",
            "Gemma 3 is chat-tuned — system prompt quality matters more than parameters.",
        ],
    },
    "phi": {
        "temp_min": 0.0, "temp_max": 0.8, "top_p_max": 0.9, "top_k_min": 10,
        "repeat_penalty_range": (1.0, 1.1), "strip_think": False,
        "notes": [
            "Phi-4 is highly instruction-tuned — keep temperature ≤ 0.7 for structured output.",
            "Avoid high top_p (>0.9); Phi models are sensitive to sampling randomness.",
            "For coding, temperature 0.1–0.3 gives the most consistent results.",
        ],
    },
    "mistral": {
        "temp_min": 0.0, "temp_max": 1.2, "top_p_max": 0.95, "top_k_min": 1,
        "repeat_penalty_range": (1.0, 1.3), "strip_think": False,
        "notes": [
            "Mistral handles top_k differently from Llama — top_k=0 (disabled) often works better.",
            "Good RAG performance at temperature 0.3–0.5 with a precise system prompt.",
        ],
    },
    "llama": {
        "temp_min": 0.0, "temp_max": 1.2, "top_p_max": 0.95, "top_k_min": 1,
        "repeat_penalty_range": (1.0, 1.3), "strip_think": False,
        "notes": ["Llama 3.x is well-balanced; most parameter changes have predictable effects."],
    },
    "qwen": {
        "temp_min": 0.0, "temp_max": 1.0, "top_p_max": 0.95, "top_k_min": 1,
        "repeat_penalty_range": (1.0, 1.2), "strip_think": False,
        "notes": ["Qwen benefits from lower temperature (0.3–0.6) for factual tasks."],
    },
    "unknown": {
        "temp_min": 0.0, "temp_max": 1.2, "top_p_max": 0.95, "top_k_min": 1,
        "repeat_penalty_range": (1.0, 1.3), "strip_think": False,
        "notes": [],
    },
}


def constraints_for(model_hint: str) -> dict:
    return FAMILY_CONSTRAINTS.get(detect_family(model_hint), FAMILY_CONSTRAINTS["unknown"])


# ── Text helpers ───────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm sorry", "i am sorry", "can't help", "cannot help",
    "i won't", "i will not", "not able to", "can't assist", "cannot assist",
    "as an ai", "i'm an ai", "i am an ai",
)


def strip_think(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks."""
    return _THINK_RE.sub("", text or "").strip()


def word_list(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", (text or "").lower())


def repetition_ratio(words: list[str], n: int = 3) -> float:
    """Fraction of repeated n-grams (0.0 = none, → 1.0 = highly repetitive)."""
    if len(words) < n * 4:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(0, len(words) - n + 1)]
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / float(len(grams))


def is_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _REFUSAL_MARKERS)

"""
AutoTune core logic (pure, UI-free, fully testable).

Defines a parameter sweep, expands it into concrete combos (capped to avoid
explosion), and scores model responses with a transparent, configurable metric.
The actual model calls live in the runner (tune/runner.py); this module never
touches the network.

Scoring (0-100, higher = better) blends:
  - errors           → hard 0
  - refusals         → heavy penalty
  - repetition       → penalty (n-gram repeat ratio)
  - length sanity    → penalty for empty/too-short; goal-aware length preference
  - keyword presence → small bonus/penalty if a required keyword is configured
  - speed            → small bonus from tokens/sec
Weights are exposed so the UI / tests can reason about them.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from . import families
from .models import PARAMETERS, coerce_param


# ── Sweep definition ─────────────────────────────────────────────────────────

# Sensible candidate values per sweepable parameter.
DEFAULT_CANDIDATES: dict[str, list] = {
    "temperature":    [0.2, 0.5, 0.8, 1.0],
    "top_p":          [0.8, 0.9, 0.95],
    "top_k":          [20, 40, 80],
    "repeat_penalty": [1.0, 1.1, 1.2],
}

# num_ctx is intentionally excluded from sweeps: changing it requires restarting
# the server, which makes a sweep slow. It stays fixed at the active value.
SWEEPABLE = tuple(DEFAULT_CANDIDATES.keys())

DEFAULT_PROMPTS = [
    "Explain what a hash map is in two sentences.",
    "Write a haiku about the ocean.",
    "List three causes of inflation, briefly.",
]

MAX_COMBOS = 24


@dataclass
class SweepSpec:
    """What to vary and what to test it against."""
    param_values: dict[str, list] = field(default_factory=dict)  # key → candidates
    base_params: dict[str, Any] = field(default_factory=dict)     # fixed params
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    system_prompt: str = ""
    goal: str = "balanced"        # balanced | accuracy | creative | concise
    keyword: str = ""             # optional required substring (case-insensitive)
    model_hint: str = ""          # for family-aware <think> stripping
    use_rag: bool = False         # inject RAG document context per prompt (runner supplies store)

    def combo_count(self) -> int:
        n = 1
        for vals in self.param_values.values():
            n *= max(1, len(vals))
        return n


def expand_grid(spec: SweepSpec, cap: int = MAX_COMBOS) -> list[dict]:
    """
    Cartesian product of the swept params, merged over base_params. Each combo
    is a full param dict. Combos beyond `cap` are dropped (caller should warn).
    """
    keys = [k for k, v in spec.param_values.items() if v]
    if not keys:
        return [dict(spec.base_params)]

    value_lists = [spec.param_values[k] for k in keys]
    combos: list[dict] = []
    for values in itertools.product(*value_lists):
        params = dict(spec.base_params)
        for k, v in zip(keys, values):
            params[k] = coerce_param(k, v)
        combos.append(params)
        if len(combos) >= cap:
            break
    return combos


# ── Scoring ──────────────────────────────────────────────────────────────────

WEIGHTS = {
    "refusal": 50.0,        # subtracted if the response refuses
    "repetition": 40.0,     # multiplied by repetition ratio, subtracted
    "too_short": 40.0,      # subtracted if response is empty/trivial
    "keyword": 10.0,        # +/- if a required keyword is set
    "speed_bonus_max": 10.0,
    "speed_ref_tps": 40.0,  # tokens/sec that earns the full speed bonus
}

_MIN_WORDS = 3


def score_response(text: str, tokens_per_second: float | None = None, *,
                   error: str | None = None, goal: str = "balanced",
                   keyword: str = "", model_hint: str = "",
                   ttft: float | None = None) -> tuple[float, dict]:
    """
    Return (score 0-100, metrics dict) for a single response.

    Transparent and monotonic: more repetition, refusals, or emptiness always
    lower the score; presence of a required keyword and higher speed raise it.
    """
    if error:
        return 0.0, {"error": error, "score": 0.0}

    fc = families.constraints_for(model_hint)
    scored = families.strip_think(text) if fc.get("strip_think") else (text or "")

    words = families.word_list(scored)
    rep = families.repetition_ratio(words)
    refusal = families.is_refusal(scored)

    # Base quality tops out at 90 so the speed bonus (≤10) has headroom under 100.
    score = 90.0
    if refusal:
        score -= WEIGHTS["refusal"]
    score -= WEIGHTS["repetition"] * rep
    if len(words) < _MIN_WORDS:
        score -= WEIGHTS["too_short"]

    # Goal-aware length nudges (mild, never dominate).
    g = (goal or "").lower()
    if g in ("concise", "short") and len(words) > 200:
        score -= 10.0
    elif g in ("creative", "creativity") and len(words) < 20:
        score -= 10.0

    if keyword:
        if keyword.lower() in scored.lower():
            score += WEIGHTS["keyword"]
        else:
            score -= WEIGHTS["keyword"]

    if tokens_per_second and tokens_per_second > 0:
        bonus = min(WEIGHTS["speed_bonus_max"],
                    WEIGHTS["speed_bonus_max"] * tokens_per_second / WEIGHTS["speed_ref_tps"])
        score += bonus

    score = max(0.0, min(100.0, score))
    metrics = {
        "score": round(score, 1),
        "words": len(words),
        "repetition": round(rep, 3),
        "refusal": refusal,
        "tokens_per_second": tokens_per_second,
        "ttft": ttft,
    }
    return score, metrics


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """One (combo, prompt) execution."""
    combo_index: int
    prompt: str
    output: str = ""
    score: float = 0.0
    metrics: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class ComboScore:
    """Aggregated score for a parameter combo across all prompts."""
    combo_index: int
    params: dict
    avg_score: float = 0.0
    runs: list[RunResult] = field(default_factory=list)
    errors: int = 0

    @property
    def swept_summary(self) -> str:
        """Compact 'temp=0.5 top_p=0.9' string of swept params only."""
        return " ".join(f"{k}={self.params[k]}" for k in SWEEPABLE if k in self.params)


def aggregate(combo_index: int, params: dict, runs: list[RunResult]) -> ComboScore:
    valid = [r for r in runs if r.error is None]
    errors = sum(1 for r in runs if r.error is not None)
    avg = round(sum(r.score for r in valid) / len(valid), 1) if valid else 0.0
    return ComboScore(combo_index=combo_index, params=dict(params),
                      avg_score=avg, runs=list(runs), errors=errors)


def rank(combos: list[ComboScore]) -> list[ComboScore]:
    """Best first: highest avg_score, fewest errors as tiebreak."""
    return sorted(combos, key=lambda c: (-c.avg_score, c.errors))

"""
Phase 5 tests: AutoTune core (tuning + families) and the runner with a fake
stream. No network.

Run:
    python tests/test_phase5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core import families, tuning
from llm_parametizer.core.tuning import SweepSpec, RunResult
from llm_parametizer.chat.engine import Backend, ChatStats
from llm_parametizer.tune import runner as runner_mod
from llm_parametizer.tune.runner import TuneRunner


# ── families ───────────────────────────────────────────────────────────────────

def test_detect_family():
    assert families.detect_family("DeepSeek-R1-7B.gguf") == "deepseek"
    assert families.detect_family("gemma-2-9b") == "gemma"
    assert families.detect_family("Qwen2.5-7B") == "qwen"
    assert families.detect_family("mixtral-8x7b") == "mistral"
    assert families.detect_family("something-else") == "unknown"


def test_constraints_for():
    fc = families.constraints_for("deepseek-r1")
    assert fc["strip_think"] is True
    assert fc["temp_max"] == 0.8


def test_strip_think():
    txt = "before <think>secret reasoning</think> after"
    assert families.strip_think(txt) == "before  after"


def test_repetition_ratio():
    none = families.word_list("the quick brown fox jumps over the lazy dog again now")
    rep = families.word_list(("spam " * 30))
    assert families.repetition_ratio(rep) > families.repetition_ratio(none)


def test_is_refusal():
    assert families.is_refusal("I'm sorry, I cannot help with that.")
    assert not families.is_refusal("Sure, here is the answer.")


# ── expand_grid ────────────────────────────────────────────────────────────────

def test_expand_grid_basic():
    spec = SweepSpec(param_values={"temperature": [0.2, 0.8], "top_p": [0.9, 0.95]},
                     base_params={"top_k": 40})
    combos = tuning.expand_grid(spec)
    assert len(combos) == 4
    # base param carried through
    assert all(c["top_k"] == 40 for c in combos)
    # swept values present
    temps = {c["temperature"] for c in combos}
    assert temps == {0.2, 0.8}


def test_expand_grid_no_sweep_returns_base():
    spec = SweepSpec(param_values={}, base_params={"temperature": 0.5})
    combos = tuning.expand_grid(spec)
    assert combos == [{"temperature": 0.5}]


def test_expand_grid_cap():
    spec = SweepSpec(param_values={
        "temperature": [0.1, 0.2, 0.3, 0.4],
        "top_p": [0.8, 0.9, 0.95],
        "top_k": [10, 20, 40],
    })
    assert spec.combo_count() == 36
    combos = tuning.expand_grid(spec, cap=10)
    assert len(combos) == 10   # capped


def test_expand_grid_coerces_types():
    spec = SweepSpec(param_values={"top_k": ["20", "40"]})
    combos = tuning.expand_grid(spec)
    assert all(isinstance(c["top_k"], int) for c in combos)


# ── scoring ────────────────────────────────────────────────────────────────────

def test_score_error_is_zero():
    score, m = tuning.score_response("anything", error="boom")
    assert score == 0.0


def test_score_refusal_penalized():
    good, _ = tuning.score_response("Here is a clear, helpful answer about hash maps and how they work.")
    bad, _ = tuning.score_response("I'm sorry, I cannot help with that request at all.")
    assert good > bad


def test_score_repetition_monotonic():
    clean, _ = tuning.score_response("The cat sat on the mat near the warm fire last night quietly.")
    spammy, _ = tuning.score_response("spam spam spam " * 20)
    assert clean > spammy


def test_score_too_short_penalized():
    full, _ = tuning.score_response("This is a complete and reasonable sentence with content.")
    tiny, _ = tuning.score_response("ok")
    assert full > tiny


def test_score_keyword_bonus():
    with_kw, _ = tuning.score_response("The capital of France is Paris.", keyword="Paris")
    without_kw, _ = tuning.score_response("The capital of France is Paris.", keyword="Berlin")
    assert with_kw > without_kw


def test_score_speed_bonus():
    slow, _ = tuning.score_response("A reasonable answer here please.", tokens_per_second=5)
    fast, _ = tuning.score_response("A reasonable answer here please.", tokens_per_second=50)
    assert fast > slow


def test_score_strip_think_for_deepseek():
    # A refusal hidden only inside <think> should not be penalized for deepseek.
    text = "<think>I cannot do this</think> Here is the helpful final answer instead."
    ds, _ = tuning.score_response(text, model_hint="deepseek-r1")
    other, _ = tuning.score_response(text, model_hint="llama")
    assert ds >= other


# ── aggregate / rank ────────────────────────────────────────────────────────────

def test_aggregate_and_rank():
    runs_a = [RunResult(0, "p", output="x", score=80.0),
              RunResult(0, "q", output="y", score=60.0)]
    runs_b = [RunResult(1, "p", output="x", score=90.0),
              RunResult(1, "q", error="fail")]
    a = tuning.aggregate(0, {"temperature": 0.5}, runs_a)
    b = tuning.aggregate(1, {"temperature": 0.8}, runs_b)
    assert a.avg_score == 70.0
    assert b.avg_score == 90.0 and b.errors == 1
    ranked = tuning.rank([a, b])
    assert ranked[0].combo_index == 1   # higher avg wins


# ── runner (fake stream) ────────────────────────────────────────────────────────

def _install_fake_stream(monkey_text="The answer is clearly forty-two and well explained here."):
    def fake_stream(backend, messages, **kw):
        # echo a deterministic answer then stats
        for tok in monkey_text.split():
            yield tok + " "
        yield ChatStats(elapsed=1.0, tokens=8, tokens_per_second=8.0)
    runner_mod.stream_chat = fake_stream  # type: ignore


def test_runner_end_to_end():
    _install_fake_stream()
    spec = SweepSpec(param_values={"temperature": [0.2, 0.8]},
                     prompts=["q1", "q2"], goal="balanced")

    finished = {}
    combos_seen = []
    progress = []

    r = TuneRunner(
        on_progress=lambda d, t, label: progress.append((d, t)),
        on_combo=lambda c: combos_seen.append(c),
        on_finished=lambda results: finished.update({"results": results}),
        on_error=lambda m: finished.update({"error": m}),
    )
    assert r.start(spec, Backend.LLAMACPP, host="x", port=1)
    r._thread.join(timeout=5)

    assert "error" not in finished
    results = finished["results"]
    assert len(results) == 2            # two combos
    assert len(combos_seen) == 2
    assert progress and progress[-1][0] == progress[-1][1]   # finished all units
    assert all(c.avg_score > 0 for c in results)


def test_runner_cancel_before_start_flag():
    _install_fake_stream()
    r = TuneRunner()
    r.cancel()
    assert r._cancel.is_set()


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)

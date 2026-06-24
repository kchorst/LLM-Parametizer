"""
Phase 8 tests: performance metrics (SessionMetrics, Snapshot, hardware_summary)
and the delivery report renderers (Markdown + plain text). No network, no UI.

Run:
    python tests/test_phase8_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core import metrics, report
from llm_parametizer.core.metrics import SessionMetrics, Snapshot
from llm_parametizer.chat.engine import ChatStats


# ── SessionMetrics ─────────────────────────────────────────────────────────────

def test_session_metrics_empty():
    sm = SessionMetrics()
    assert sm.count == 0
    assert sm.total_tokens == 0
    assert sm.avg_tps is None
    assert sm.avg_ttft is None
    assert sm.summary()["messages"] == 0


def test_session_metrics_accumulates():
    sm = SessionMetrics()
    sm.add(ChatStats(elapsed=2.0, tokens=100, tokens_per_second=50.0, ttft=0.3))
    sm.add(ChatStats(elapsed=4.0, tokens=200, tokens_per_second=50.0, ttft=0.5))
    assert sm.count == 2
    assert sm.total_tokens == 300
    assert sm.avg_tps == 50.0
    assert sm.avg_ttft == 0.4
    assert sm.min_ttft == 0.3
    assert sm.max_ttft == 0.5


def test_session_metrics_ignores_errors_and_empty():
    sm = SessionMetrics()
    sm.add(ChatStats(error="boom"))
    sm.add(ChatStats(tokens=0))
    sm.add(None)
    assert sm.count == 0


def test_session_metrics_handles_missing_ttft():
    sm = SessionMetrics()
    sm.add(ChatStats(elapsed=1.0, tokens=10, tokens_per_second=10.0, ttft=None))
    assert sm.count == 1
    assert sm.avg_ttft is None
    assert sm.avg_tps == 10.0


def test_session_metrics_reset():
    sm = SessionMetrics()
    sm.add(ChatStats(elapsed=1.0, tokens=10, tokens_per_second=10.0, ttft=0.1))
    sm.reset()
    assert sm.count == 0
    assert sm.total_tokens == 0
    assert sm.avg_tps is None


# ── Snapshot ────────────────────────────────────────────────────────────────────

def test_snapshot_metric_lookup():
    s = Snapshot(label="X", perf={"avg_tps": 42.0}, quality={"score": 88.0})
    assert s.metric("avg_tps") == 42.0
    assert s.metric("score") == 88.0
    assert s.metric("missing") is None


def test_hardware_summary_keys():
    hw = metrics.hardware_summary()
    for key in ("os", "cpu", "ram_total_gb", "gpu"):
        assert key in hw


# ── Report rendering ─────────────────────────────────────────────────────────────

def _snap(label, tps, ttft, score=None):
    perf = {"avg_tps": tps, "avg_ttft": ttft, "total_tokens": 300}
    quality = {"score": score} if score is not None else {}
    return Snapshot(label=label, model_name="Qwen2.5-7B-Q4.gguf", goal="accuracy",
                    params={"temperature": 0.4, "top_p": 0.9},
                    command="llama-server --model x", perf=perf, quality=quality)


def test_report_empty():
    md = report.build_markdown([])
    assert "No configuration" in md


def test_report_single_markdown():
    md = report.build_markdown([_snap("Baseline", 30.0, 0.8)])
    assert "# LLM Parametizer" in md
    assert "Qwen2.5-7B-Q4.gguf" in md
    assert "Tokens / second" in md
    assert "`temperature`" in md


def test_report_before_after_has_delta():
    before = _snap("Baseline", 30.0, 0.8)
    after = _snap("Optimized", 45.0, 0.5)
    md = report.build_markdown([before, after])
    assert "Baseline" in md and "Optimized" in md
    # TPS improved (higher better) and TTFT improved (lower better)
    assert "better" in md
    assert "+50%" in md or "+50" in md


def test_report_plaintext_renders():
    txt = report.build_text([_snap("Baseline", 30.0, 0.8)], client="Acme")
    assert "MODEL DELIVERY REPORT" in txt
    assert "Acme" in txt
    assert "RECOMMENDED PARAMETERS" in txt


def test_report_family_notes():
    notes = report.family_notes("DeepSeek-R1-7B.gguf", goal="accuracy")
    assert any("deepseek" in n.lower() for n in notes)


def test_delta_direction_worse():
    # TPS dropping is worse
    s = report._delta(50.0, 30.0, higher_is_better=True)
    assert "worse" in s
    # TTFT rising is worse
    s2 = report._delta(0.3, 0.6, higher_is_better=False)
    assert "worse" in s2


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

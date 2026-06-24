"""
Phase 7 tests: backend log ring buffer + accessors on BackendController.
No process is spawned; we exercise the capture/forward/clear logic directly.

Run:
    python tests/test_phase7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core.config import Settings
from llm_parametizer.backend.controller import BackendController


def _controller(on_log=None):
    return BackendController(Settings(), on_log=on_log)


def test_log_capture_and_text():
    c = _controller()
    assert c.log_line_count() == 0
    c._capture_log("hello")
    c._capture_log("world")
    assert c.log_line_count() == 2
    assert c.log_text() == "hello\nworld"


def test_log_forwards_to_user_callback():
    seen = []
    c = _controller(on_log=seen.append)
    c._capture_log("line-1")
    assert seen == ["line-1"]


def test_log_user_callback_error_is_swallowed():
    def boom(_l):
        raise ValueError("nope")
    c = _controller(on_log=boom)
    # Must not raise even if the UI callback fails.
    c._capture_log("safe")
    assert c.log_line_count() == 1


def test_log_clear():
    c = _controller()
    c._capture_log("a")
    c._capture_log("b")
    c.clear_log()
    assert c.log_line_count() == 0
    assert c.log_text() == ""


def test_log_ring_buffer_caps():
    c = _controller()
    # maxlen is 5000; push more and confirm it caps.
    for i in range(5200):
        c._capture_log(str(i))
    assert c.log_line_count() == 5000
    # Oldest dropped, newest retained.
    assert c.log_text().splitlines()[0] == "200"
    assert c.log_text().splitlines()[-1] == "5199"


def test_pid_none_when_not_running():
    c = _controller()
    assert c.pid is None
    assert c.is_running is False


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

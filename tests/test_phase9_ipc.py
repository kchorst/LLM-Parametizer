"""
Phase 9 tests: IPC bridge compatibility with the Sentinel+ / LLM Tester
ecosystem. Exercises the dispatch allow-list, the safe-status provider, the
hardened set_config validation, and profile sanitisation — no real sockets.

Run:
    python tests/test_phase9_ipc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core import ipc
from llm_parametizer.ui.app import LLMParametizerApp


# ── Allow-list / dispatch (no app needed) ────────────────────────────────────

def test_required_actions_allowed():
    required = {"ping", "get_status", "get_model_info", "get_config",
                "set_config", "export_profile", "shield_heartbeat"}
    assert required <= ipc.ALLOWED_ACTIONS


def test_unknown_action_rejected():
    status, body = ipc._dispatch("definitely_not_allowed", {})
    assert status == 403
    assert "not permitted" in body["error"]


def test_register_handler_rejects_unlisted():
    try:
        ipc.register_handler("bogus_action", lambda d: {})
        assert False, "should have raised"
    except ValueError:
        pass


def test_ping_handler():
    status, body = ipc._dispatch("ping", {"action": "ping"})
    assert status == 200
    assert body["result"]["app"] == "llm_parametizer"


def test_bind_is_localhost_only():
    # The server must only ever bind to 127.0.0.1 (rule 3).
    src = (Path(__file__).resolve().parent.parent
           / "llm_parametizer" / "core" / "ipc.py").read_text(encoding="utf-8")
    assert '"127.0.0.1"' in src
    assert '"0.0.0.0"' not in src


# ── Profile sanitisation (pure, classmethod) ─────────────────────────────────

def test_safe_profile_strips_unsafe_fields():
    raw = {
        "config_name": "fast",
        "temperature": 0.4,
        "system_prompt": "secret instructions",
        "template": "{{prompt}}",
        "gguf_path": "C:/models/x.gguf",
        "llama_server_path": "C:/llama/server.exe",
        "api_key": "sk-123",
        "extra_llama_args": ["--foo"],
    }
    safe = LLMParametizerApp._safe_profile(raw)
    assert "system_prompt" not in safe
    assert "template" not in safe
    assert "gguf_path" not in safe
    assert "llama_server_path" not in safe
    assert "api_key" not in safe
    assert "extra_llama_args" not in safe
    # Safe sampling/metadata survives.
    assert safe["config_name"] == "fast"
    assert safe["temperature"] == 0.4


# ── App-level handlers via a headless app ─────────────────────────────────────

_APP = None


def _app():
    """Construct the Tk app once; skip these tests if no display is available."""
    global _APP
    if _APP is None:
        _APP = LLMParametizerApp()
        _APP.withdraw()
    return _APP


def test_safe_status_has_required_fields():
    app = _app()
    s = app._safe_status()
    for key in ("running", "reachable", "engine", "model", "profileName",
                "params", "updatedAt"):
        assert key in s, f"missing {key}"
    # Must not leak raw text or paths.
    assert "system_prompt" not in s
    assert "template" not in s


def test_safe_status_vram_only_when_detected():
    app = _app()
    s = app._safe_status()
    from llm_parametizer.core import memory
    detected = memory.vram_info().available
    assert ("vram" in s) == bool(detected and memory.vram_info().total_bytes)


def test_set_config_accepts_known_and_coerces():
    app = _app()
    handlers = {}
    import llm_parametizer.core.ipc as ipc_mod
    # Re-register to capture the closure handlers in a fresh dict.
    orig = ipc_mod.register_handler
    ipc_mod.register_handler = lambda a, h: handlers.__setitem__(a, h)
    try:
        app._register_ipc_handlers()
    finally:
        ipc_mod.register_handler = orig
    res = handlers["set_config"]({"params": {"temperature": "0.3", "top_k": 999}})
    assert res["applied"]["temperature"] == 0.3          # coerced str -> float
    assert res["applied"]["top_k"] == 100                # clamped to max
    assert app.active.params["temperature"] == 0.3


def test_set_config_rejects_unknown_param():
    app = _app()
    handlers = {}
    import llm_parametizer.core.ipc as ipc_mod
    orig = ipc_mod.register_handler
    ipc_mod.register_handler = lambda a, h: handlers.__setitem__(a, h)
    try:
        app._register_ipc_handlers()
    finally:
        ipc_mod.register_handler = orig
    res = handlers["set_config"]({"params": {"evil_key": 1, "temperature": 0.5}})
    assert "evil_key" in res["rejected"]
    assert "temperature" in res["applied"]
    assert "evil_key" not in app.active.params


def _capture_handlers(app):
    handlers = {}
    import llm_parametizer.core.ipc as ipc_mod
    orig = ipc_mod.register_handler
    ipc_mod.register_handler = lambda a, h: handlers.__setitem__(a, h)
    try:
        app._register_ipc_handlers()
    finally:
        ipc_mod.register_handler = orig
    return handlers


def test_report_metrics_action_allowed():
    assert "report_metrics" in ipc.ALLOWED_ACTIONS
    assert "get_metrics" in ipc.ALLOWED_ACTIONS


def test_report_metrics_stores_safe_numbers():
    app = _app()
    h = _capture_handlers(app)
    res = h["report_metrics"]({"metrics": {
        "source": "LLM Tester", "modelName": "qwen2.5", "tps": "45.2",
        "ttftMs": "320ms", "totalTokens": 256, "status": "success",
        "prompt": "should be ignored", "response": "should be ignored",
    }})
    assert res["stored"] is True
    stored = app._last_test_metrics
    assert stored["tps"] == 45.2
    assert stored["ttftMs"] == 320
    assert stored["model"] == "qwen2.5"
    # Raw text must never be stored.
    assert "prompt" not in stored and "response" not in stored


def test_get_metrics_and_status_reflect_report():
    app = _app()
    h = _capture_handlers(app)
    h["report_metrics"]({"metrics": {"source": "LLM Tester", "tps": 30, "ttftMs": 500}})
    got = h["get_metrics"]({})
    assert got["lastTestMetrics"]["tps"] == 30.0
    assert app._safe_status()["lastTestMetrics"]["ttftMs"] == 500.0


def test_get_model_info_hides_system_prompt():
    app = _app()
    handlers = {}
    import llm_parametizer.core.ipc as ipc_mod
    orig = ipc_mod.register_handler
    ipc_mod.register_handler = lambda a, h: handlers.__setitem__(a, h)
    try:
        app._register_ipc_handlers()
    finally:
        ipc_mod.register_handler = orig
    info = handlers["get_model_info"]({})
    assert "system_prompt" not in info
    assert "template" not in info
    assert "params" in info


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
    # Clean up the Tk app if it was created.
    global _APP
    if _APP is not None:
        try:
            _APP.destroy()
        except Exception:
            pass
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)

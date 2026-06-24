"""
Phase 3 controller tests (no UI).

Run:
    python tests/test_phase3_controller.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core.config import Settings
from llm_parametizer.chat.engine import Backend
from llm_parametizer.backend.controller import (
    BackendController, ENGINE_LLAMA_SERVER, ENGINE_OLLAMA, ENGINE_LLAMA_SWAP,
)


def _settings() -> Settings:
    d = tempfile.mkdtemp()
    return Settings(Path(d) / "s.json")


def test_default_engine_is_llamacpp():
    c = BackendController(_settings())
    assert c.engine == ENGINE_LLAMA_SERVER
    assert c.is_ollama is False
    assert c.has_port is True


def test_chat_target_llamacpp():
    s = _settings()
    s["llama_host"] = "127.0.0.1"
    s["llama_port"] = 8080
    c = BackendController(s)
    backend, host, port, model = c.chat_target()
    assert backend == Backend.LLAMACPP
    assert host == "127.0.0.1"
    assert port == 8080


def test_chat_target_ollama():
    s = _settings()
    s["default_backend"] = ENGINE_OLLAMA
    c = BackendController(s)
    backend, host, port, model = c.chat_target()
    assert backend == Backend.OLLAMA
    assert port == 11434


def test_chat_target_swap_parses_listen():
    s = _settings()
    s["default_backend"] = ENGINE_LLAMA_SWAP
    s["swap_listen"] = "127.0.0.1:8090"
    c = BackendController(s)
    backend, host, port, _ = c.chat_target()
    assert backend == Backend.LLAMACPP
    assert host == "127.0.0.1"
    assert port == 8090


def test_build_config_uses_params():
    c = BackendController(_settings())
    c.set_params({"temperature": 0.33, "num_ctx": 8192})
    c.set_gguf_path("C:/m/x.gguf")
    cfg = c.build_config()
    assert cfg.temperature == 0.33
    assert cfg.num_ctx == 8192
    assert cfg.gguf_path == "C:/m/x.gguf"


def test_start_ollama_noop():
    s = _settings()
    s["default_backend"] = ENGINE_OLLAMA
    c = BackendController(s)
    ok, msg = c.start()
    assert ok is True
    assert "external" in msg.lower()


def test_start_rejects_missing_model():
    c = BackendController(_settings())
    c.set_gguf_path("")
    ok, msg = c.start()
    assert ok is False
    assert "model" in msg.lower()


def test_start_rejects_nonexistent_model():
    c = BackendController(_settings())
    c.set_gguf_path("C:/does/not/exist.gguf")
    ok, msg = c.start()
    assert ok is False
    assert "not found" in msg.lower()


def test_start_rejects_missing_binary():
    s = _settings()
    c = BackendController(s)
    # Real model file, but no llama-server binary configured/findable
    with tempfile.TemporaryDirectory() as d:
        model = Path(d) / "m.gguf"
        model.write_bytes(b"x" * 1000)
        c.set_gguf_path(str(model))
        ok, msg = c.start(skip_preflight=True)
        # Either binary missing OR (if user has llama-server on PATH) it tries to start.
        # In CI/dev without llama-server, expect binary-not-found.
        if not ok:
            assert "binary not found" in msg.lower()


def test_command_preview_llamacpp():
    c = BackendController(_settings())
    c.set_gguf_path("C:/m/x.gguf")
    preview = c.command_preview()
    assert "x.gguf" in preview


def test_chat_reachable_false_when_down():
    s = _settings()
    s["llama_port"] = 59994
    c = BackendController(s)
    assert c.chat_reachable() is False


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

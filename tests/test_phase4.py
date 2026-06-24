"""
Phase 4 tests: profile library + ActiveConfig shared state.

Run:
    python tests/test_phase4.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core.models import ModelConfig, default_params
from llm_parametizer.core.state import ActiveConfig
from llm_parametizer.core.library import ProfileLibrary, sanitize_name


# ── ProfileLibrary ─────────────────────────────────────────────────────────────

def _lib() -> ProfileLibrary:
    return ProfileLibrary(tempfile.mkdtemp())


def test_save_load_roundtrip():
    lib = _lib()
    cfg = ModelConfig(gguf_path="C:/m/x.gguf", temperature=0.42, num_ctx=8192)
    assert lib.save("my-profile", cfg) is True
    assert "my-profile" in lib.list_profiles()
    loaded = lib.load("my-profile")
    assert loaded is not None
    assert loaded.temperature == 0.42
    assert loaded.num_ctx == 8192
    assert loaded.config_name == "my-profile"


def test_load_missing_returns_none():
    lib = _lib()
    assert lib.load("nope") is None


def test_delete():
    lib = _lib()
    lib.save("p", ModelConfig())
    assert lib.delete("p") is True
    assert "p" not in lib.list_profiles()
    assert lib.delete("p") is False


def test_rename():
    lib = _lib()
    lib.save("old", ModelConfig(temperature=0.7))
    assert lib.rename("old", "new") is True
    assert "old" not in lib.list_profiles()
    assert "new" in lib.list_profiles()
    cfg = lib.load("new")
    assert cfg.temperature == 0.7
    assert cfg.config_name == "new"


def test_rename_to_existing_fails():
    lib = _lib()
    lib.save("a", ModelConfig())
    lib.save("b", ModelConfig())
    assert lib.rename("a", "b") is False


def test_duplicate():
    lib = _lib()
    lib.save("base", ModelConfig(top_k=55))
    assert lib.duplicate("base", "copy") is True
    assert lib.load("copy").top_k == 55
    # duplicating onto existing fails
    assert lib.duplicate("base", "copy") is False


def test_sanitize_name():
    assert sanitize_name("a/b:c*?") == "a_b_c_"
    assert sanitize_name("   ") == "untitled"
    assert sanitize_name("Good Name-1.2") == "Good Name-1.2"


# ── ActiveConfig ───────────────────────────────────────────────────────────────

def test_active_defaults():
    a = ActiveConfig()
    assert a.params == default_params()
    assert a.system_prompt == ""


def test_active_set_param_notifies():
    a = ActiveConfig()
    hits = []
    a.subscribe(lambda: hits.append(1))
    a.set_param("temperature", 0.5)
    assert a.params["temperature"] == 0.5
    assert hits  # observer fired


def test_active_set_param_no_notify():
    a = ActiveConfig()
    hits = []
    a.subscribe(lambda: hits.append(1))
    a.set_param("top_k", 33, notify=False)
    assert a.params["top_k"] == 33
    assert not hits


def test_active_coerces():
    a = ActiveConfig()
    a.set_param("top_k", "40", notify=False)
    assert a.params["top_k"] == 40 and isinstance(a.params["top_k"], int)


def test_active_load_from_cfg():
    a = ActiveConfig()
    cfg = ModelConfig(temperature=0.25, system_prompt="hi", config_name="prof")
    a.load_from(cfg, notify=False)
    assert a.params["temperature"] == 0.25
    assert a.system_prompt == "hi"
    assert a.profile_name == "prof"


def test_active_to_model_config():
    a = ActiveConfig()
    a.set_param("num_ctx", 2048, notify=False)
    a.set_system_prompt("sys", notify=False)
    cfg = a.to_model_config(gguf_path="C:/m/x.gguf", port=9000)
    assert cfg.num_ctx == 2048
    assert cfg.system_prompt == "sys"
    assert cfg.gguf_path == "C:/m/x.gguf"
    assert cfg.port == 9000


def test_active_reset():
    a = ActiveConfig()
    a.set_param("temperature", 1.9, notify=False)
    a.reset_params(notify=False)
    assert a.params["temperature"] == default_params()["temperature"]


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

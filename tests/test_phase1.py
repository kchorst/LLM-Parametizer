"""
Phase 1 foundation tests: config, models, paths, memory.

Run from the project root:
    python -m pytest tests/test_phase1.py -v
or without pytest:
    python tests/test_phase1.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core.config import Settings, DEFAULT_SETTINGS
from llm_parametizer.core.models import (
    ModelConfig, PARAMETERS, default_params, coerce_param,
)
from llm_parametizer.core import paths
from llm_parametizer.core import memory


# ── Config ─────────────────────────────────────────────────────────────────────

def test_settings_defaults():
    with tempfile.TemporaryDirectory() as d:
        s = Settings(Path(d) / "settings.json")
        assert s.get("default_backend") == "llama.cpp (server)"
        assert s.get_int("llama_port") == 8080
        assert s.get_bool("flash_attn") is False


def test_settings_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "settings.json"
        s = Settings(path)
        s["gguf_models_dir"] = "C:/models"
        s["llama_port"] = 9090
        assert s.save() is True

        s2 = Settings(path)
        assert s2.get("gguf_models_dir") == "C:/models"
        assert s2.get_int("llama_port") == 9090


def test_settings_typed_getters():
    with tempfile.TemporaryDirectory() as d:
        s = Settings(Path(d) / "settings.json")
        s["x"] = "true"
        assert s.get_bool("x") is True
        s["y"] = "12"
        assert s.get_int("y") == 12
        s["z"] = "1.5"
        assert s.get_float("z") == 1.5


# ── Models ─────────────────────────────────────────────────────────────────────

def test_default_params():
    p = default_params()
    assert p["temperature"] == 0.8
    assert p["num_ctx"] == 4096
    assert set(p.keys()) == set(PARAMETERS.keys())


def test_coerce_param():
    assert coerce_param("top_k", "40") == 40
    assert isinstance(coerce_param("temperature", "0.7"), float)
    # Bad value falls back to default
    assert coerce_param("top_k", "abc") == PARAMETERS["top_k"]["default"]


def test_modelconfig_from_params():
    cfg = ModelConfig.from_params(
        {"temperature": 0.5, "num_ctx": 8192},
        gguf_path="C:/models/foo.gguf",
    )
    assert cfg.temperature == 0.5
    assert cfg.num_ctx == 8192
    assert cfg.gguf_path == "C:/models/foo.gguf"
    assert cfg.model_name == "foo"


def test_modelconfig_dict_roundtrip():
    cfg = ModelConfig(gguf_path="x/y/model.gguf", temperature=0.3)
    d = cfg.to_dict()
    cfg2 = ModelConfig.from_dict(d)
    assert cfg2.gguf_path == cfg.gguf_path
    assert cfg2.temperature == 0.3
    # Unknown keys ignored
    cfg3 = ModelConfig.from_dict({"gguf_path": "a.gguf", "bogus": 1})
    assert cfg3.gguf_path == "a.gguf"


def test_modelconfig_params_subset():
    cfg = ModelConfig(temperature=0.9, top_k=50)
    p = cfg.params()
    assert p["temperature"] == 0.9
    assert p["top_k"] == 50
    assert "gguf_path" not in p


# ── Paths ──────────────────────────────────────────────────────────────────────

def test_resolve_binary_explicit(tmp_path=None):
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
        fake.write_text("")
        resolved = paths.resolve_binary(str(fake), "", "llama-server")
        assert resolved == str(fake)


def test_resolve_binary_bindir():
    with tempfile.TemporaryDirectory() as d:
        name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        (Path(d) / name).write_text("")
        resolved = paths.resolve_llama_server("", d)
        assert Path(resolved).name == name


def test_resolve_binary_missing():
    assert paths.resolve_binary("", "", "nonexistent-binary-xyz") == ""


def test_scan_gguf_empty():
    assert paths.scan_gguf_models("") == []
    assert paths.scan_gguf_models("C:/this/does/not/exist") == []


def test_scan_gguf_finds_files():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.gguf").write_bytes(b"x" * 100)
        sub = Path(d) / "sub"
        sub.mkdir()
        (sub / "b.gguf").write_bytes(b"y" * 200)
        entries = paths.scan_gguf_models(d)
        names = {e.stem for e in entries}
        assert names == {"a", "b"}


# ── Memory ─────────────────────────────────────────────────────────────────────

def test_system_memory():
    total, avail = memory.system_memory()
    # Either psutil works (positive) or returns (0,0)
    assert total >= 0 and avail >= 0
    if total:
        assert avail <= total


def test_estimate_model_ram_no_file():
    est = memory.estimate_model_ram("", 4096)
    assert est.model_bytes == 0
    assert est.kv_bytes > 0
    assert est.total_bytes == est.kv_bytes


def test_estimate_scales_with_ctx():
    e1 = memory.estimate_model_ram("", 2048)
    e2 = memory.estimate_model_ram("", 8192)
    assert e2.kv_bytes > e1.kv_bytes


def test_preflight_unreadable_is_permissive(monkeypatch=None):
    # Force system_memory to report unreadable
    orig = memory.system_memory
    memory.system_memory = lambda: (0, 0)  # type: ignore
    try:
        r = memory.preflight_check("", 4096)
        assert r.fits is True
    finally:
        memory.system_memory = orig  # type: ignore


def test_flush_memory_returns_int():
    n = memory.flush_memory()
    assert isinstance(n, int)


# ── Manual runner ───────────────────────────────────────────────────────────────

def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
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

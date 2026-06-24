"""
Phase 2 tests: backend (process, llama_cpp, ollama) + chat (history, engine, session).

Run:
    python tests/test_phase2.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.core.models import ModelConfig, default_params
from llm_parametizer.backend import process as pm
from llm_parametizer.backend import llama_cpp as lc
from llm_parametizer.backend import ollama
from llm_parametizer.chat.history import ChatHistory, Message
from llm_parametizer.chat.engine import Backend, ChatStats, stream_chat
from llm_parametizer.chat.session import ChatSession


# ── llama_cpp adapter ─────────────────────────────────────────────────────────

def test_build_argv_server():
    cfg = ModelConfig(gguf_path="C:/m/x.gguf", num_ctx=8192, temperature=0.5,
                      llama_server_path="llama-server")
    argv = lc.build_argv(cfg, mode="server")
    assert "--model" in argv and "C:/m/x.gguf" in argv
    assert "--ctx-size" in argv and "8192" in argv
    assert "--temp" in argv and "0.5" in argv
    assert "--port" in argv


def test_build_argv_omits_default_seed_and_npredict():
    cfg = ModelConfig(gguf_path="x.gguf", seed=0, num_predict=-1)
    argv = lc.build_argv(cfg, mode="server")
    assert "--seed" not in argv
    assert "--n-predict" not in argv


def test_ini_roundtrip():
    cfg = ModelConfig(gguf_path="C:/m/model.gguf", temperature=0.3, num_ctx=8192,
                      system_prompt="line1\nline2", n_gpu_layers=10, port=9999)
    text = lc.to_ini(cfg)
    cfg2 = lc.from_ini(text)
    assert cfg2.gguf_path == cfg.gguf_path
    assert cfg2.temperature == 0.3
    assert cfg2.num_ctx == 8192
    assert cfg2.system_prompt == "line1\nline2"   # multiline survives
    assert cfg2.n_gpu_layers == 10
    assert cfg2.port == 9999


def test_from_ini_tolerates_garbage():
    cfg = lc.from_ini("][not ini @@@ {{{")
    assert isinstance(cfg, ModelConfig)
    # Falls back to defaults
    assert cfg.temperature == default_params()["temperature"]


def test_to_ini_embeds_command():
    cfg = ModelConfig(gguf_path="C:/m/model.gguf", llama_server_path="llama-server")
    text = lc.to_ini(cfg)
    assert "llama-server" in text
    assert "--model" in text
    assert "[command]" in text


def test_to_modelfile_basic():
    cfg = ModelConfig(ollama_base_model="llama3", temperature=0.5, system_prompt="hi")
    mf = lc.to_modelfile(cfg)
    assert mf.startswith("FROM llama3")
    assert "SYSTEM" in mf
    assert "PARAMETER temperature 0.5" in mf


def test_swap_config():
    cfg = ModelConfig(gguf_path="C:/m/x.gguf", swap_listen="127.0.0.1:8090")
    sc = lc.build_swap_config(cfg)
    assert sc["listen"] == "127.0.0.1:8090"
    assert sc["models"] and sc["models"][0]["name"] == "x"


# ── ollama client (offline-safe) ────────────────────────────────────────────────

def test_ollama_base_url():
    assert ollama.base_url("h", 1234) == "http://h:1234"


def test_ollama_available_offline():
    # Unlikely to be running on this port; must return False, not raise
    assert ollama.is_available("127.0.0.1", 59999) is False


def test_ollama_list_models_offline():
    assert ollama.list_models("127.0.0.1", 59999) == []


# ── process manager ──────────────────────────────────────────────────────────────

def test_port_open_false():
    assert pm.port_open("127.0.0.1", 59998, timeout=0.2) is False


def test_process_lifecycle_cli():
    """Start a short-lived python process in CLI mode; verify clean state after."""
    statuses = []
    mgr = pm.ProcessManager(on_status=statuses.append)
    argv = [sys.executable, "-c", "print('hello'); import sys; sys.exit(0)"]
    mgr.start(argv, mode="cli")
    # Wait for it to finish
    deadline = time.time() + 5
    while mgr.is_running and time.time() < deadline:
        time.sleep(0.05)
    assert mgr.is_running is False
    # _proc must be cleared
    assert mgr.pid is None
    assert "starting" in statuses


def test_process_clears_on_error_exit():
    """Non-zero exit must clear _proc and emit 'error'."""
    err_seen = threading.Event()
    statuses = []

    def on_status(s):
        statuses.append(s)
        if s == "error":
            err_seen.set()

    mgr = pm.ProcessManager(on_status=on_status)
    argv = [sys.executable, "-c", "import sys; sys.exit(3)"]
    mgr.start(argv, mode="cli")
    # Wait specifically for the log thread to finalize with 'error'
    assert err_seen.wait(timeout=5), f"no error status; saw {statuses}"
    assert mgr.is_running is False
    assert mgr.pid is None


def test_process_rejects_double_start():
    mgr = pm.ProcessManager()
    argv = [sys.executable, "-c", "import time; time.sleep(2)"]
    mgr.start(argv, mode="cli")
    raised = False
    try:
        mgr.start(argv, mode="cli")
    except RuntimeError:
        raised = True
    mgr.stop()
    assert raised is True


# ── chat history ─────────────────────────────────────────────────────────────────

def test_history_add_and_clear():
    h = ChatHistory()
    h.add_user("hi")
    h.add_assistant("hello")
    assert len(h) == 2
    h.clear()
    assert h.is_empty()


def test_history_payload_includes_system():
    h = ChatHistory()
    h.add_user("question")
    payload = h.build_payload(system_prompt="sys", num_ctx=4096)
    assert payload[0] == {"role": "system", "content": "sys"}
    assert payload[-1]["content"] == "question"


def test_history_trims_to_budget():
    h = ChatHistory()
    # Add many long messages
    for i in range(50):
        h.add_user("x" * 500)
        h.add_assistant("y" * 500)
    payload = h.build_payload(system_prompt="", num_ctx=2048, budget_fraction=0.5)
    # Budget ~ 2048*0.5*4 = 4096 chars; should keep only a few recent messages
    total = sum(len(m["content"]) for m in payload)
    assert total <= 4096 or len(payload) <= 4
    assert len(payload) >= 2  # never trims below 2


def test_history_max_turns_cap():
    h = ChatHistory(max_turns=10)
    for i in range(30):
        h.add_user(str(i))
    assert len(h) <= 10


def test_history_transcript():
    h = ChatHistory()
    h.add_user("hi")
    h.add_assistant("yo")
    t = h.transcript()
    assert "You:" in t and "Assistant:" in t


# ── engine error paths (offline) ─────────────────────────────────────────────────

def test_stream_llamacpp_connection_error():
    items = list(stream_chat(Backend.LLAMACPP, [{"role": "user", "content": "hi"}],
                             host="127.0.0.1", port=59997))
    assert len(items) == 1
    assert isinstance(items[0], ChatStats)
    assert items[0].error is not None


def test_stream_ollama_requires_model():
    items = list(stream_chat(Backend.OLLAMA, [{"role": "user", "content": "hi"}],
                             host="127.0.0.1", port=59997, model=""))
    assert isinstance(items[0], ChatStats)
    assert "model" in (items[0].error or "").lower()


# ── session (offline error path) ─────────────────────────────────────────────────

def test_session_send_error_callback():
    errors = []
    done = threading.Event()
    sess = ChatSession(
        on_error=lambda m: (errors.append(m), done.set()),
    )
    started = sess.send("hi", Backend.LLAMACPP, host="127.0.0.1", port=59996)
    assert started is True
    assert done.wait(timeout=5)
    assert errors and errors[0]
    # Busy lock released
    assert sess.is_busy is False


def test_session_rejects_empty():
    sess = ChatSession()
    assert sess.send("   ", Backend.LLAMACPP) is False


def test_session_user_message_recorded():
    done = threading.Event()
    sess = ChatSession(on_error=lambda m: done.set())
    sess.send("remember me", Backend.LLAMACPP, host="127.0.0.1", port=59995)
    done.wait(timeout=5)
    # The user message should be in history even though generation failed
    assert any(m.role == "user" and m.content == "remember me" for m in sess.history.messages)


# ── runner ─────────────────────────────────────────────────────────────────────

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

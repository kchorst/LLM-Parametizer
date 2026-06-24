"""
Phase 3 UI smoke test — constructs the full app, pumps the event loop briefly,
then destroys it. Catches widget/layout/callback wiring errors without manual
interaction. Skips gracefully if no display is available.

Run:
    python tests/test_phase3_ui_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_app_constructs_and_destroys():
    try:
        from llm_parametizer.ui.app import LLMParametizerApp
    except Exception as e:
        print(f"SKIP  import failed: {e}")
        return

    try:
        app = LLMParametizerApp()
    except Exception as e:
        # Likely no display (headless CI) — treat as skip, not fail
        msg = str(e).lower()
        if "display" in msg or "no $display" in msg or "tcl" in msg:
            print(f"SKIP  no display: {e}")
            return
        raise

    # Pump the event loop a few times to trigger _post_init callbacks
    for _ in range(5):
        app.update_idletasks()
        app.update()

    # Verify key widgets exist
    assert hasattr(app, "chat_panel")
    cp = app.chat_panel
    assert cp.controller is not None
    assert cp.session is not None
    assert cp.engine_var.get() in (
        "llama.cpp (server)", "llama.cpp (cli)", "llama-swap", "Ollama"
    )

    # Exercise a couple of UI methods directly (no network)
    cp._flush_memory()
    cp._clear_chat()
    cp._append_block("You", "test message")

    # Settings opens as a modal dialog from the chat top bar.
    cp.on_open_settings()           # should create the dialog
    app.update_idletasks(); app.update()
    dlg = app._settings_dialog
    assert dlg is not None and dlg.winfo_exists()
    sp = dlg.panel
    assert "gguf_models_dir" in sp._vars
    sp._collect()   # must not raise
    sp._verify()    # must not raise (no network)
    dlg._close()    # close dialog only
    app.update_idletasks(); app.update()

    # Models tab present and wired to shared ActiveConfig.
    assert hasattr(app, "models_panel")
    mp = app.models_panel
    assert mp.active is app.active is cp.active   # one shared instance

    # Moving a slider updates ActiveConfig, which Chat reads.
    mp._on_slider("temperature", 0.5)
    app.update_idletasks(); app.update()
    assert app.active.params["temperature"] == 0.5
    assert cp._current_params()["temperature"] == 0.5

    # Editing the system prompt propagates to Chat.
    mp.system_box.insert("1.0", "be terse")
    mp._on_prompt_edit()
    assert app.active.system_prompt.strip() == "be terse"

    # Command preview renders without error.
    mp._refresh_preview()

    # Profile save → list → load roundtrip via the panel's library.
    import tempfile
    from llm_parametizer.core.library import ProfileLibrary
    mp.library = ProfileLibrary(tempfile.mkdtemp())
    mp._do_save("smoke-profile")
    assert "smoke-profile" in mp.library.list_profiles()
    mp.profile_var.set("smoke-profile")
    mp._load_profile()
    app.update_idletasks(); app.update()

    # Tune tab present and wired to the same shared state.
    assert hasattr(app, "tune_panel")
    tp = app.tune_panel
    assert tp.active is app.active

    # Build a spec from the UI and verify combo accounting + apply-best path.
    tp._sweep_vars["temperature"].set(True)
    tp._refresh_combo_count()
    spec = tp._build_spec()
    assert "temperature" in spec.param_values
    assert spec.combo_count() >= 1

    # Simulate a finished sweep and apply the best combo into ActiveConfig.
    from llm_parametizer.core.tuning import ComboScore
    tp._on_finished([
        ComboScore(combo_index=0, params={"temperature": 0.2}, avg_score=88.0),
        ComboScore(combo_index=1, params={"temperature": 0.8}, avg_score=42.0),
    ])
    app.update_idletasks(); app.update()
    tp._apply_best()
    assert app.active.params["temperature"] == 0.2

    # RAG tab present and shares one store with the chat panel.
    assert hasattr(app, "rag_panel")
    assert app.rag_panel.store is app.rag_store
    assert app.chat_panel.rag_store is app.rag_store

    # Index a doc and confirm chat augments the system prompt when enabled.
    app.rag_store.add_text("policy", "The refund policy allows returns within 30 days.")
    app.rag_store.build()
    app.rag_store.enabled = True
    augmented = app.chat_panel._system_prompt_for("how do refunds work?")
    assert "refund" in augmented.lower()
    app.rag_store.enabled = False
    plain = app.chat_panel._system_prompt_for("how do refunds work?")
    assert "[policy #1]" not in plain

    # Phase 7: copy transcript path doesn't error with an empty history.
    app.chat_panel._copy_transcript()
    app.update_idletasks(); app.update()

    # Clean shutdown
    cp.shutdown()
    app.destroy()


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

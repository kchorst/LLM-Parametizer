# LLM Parametizer — Session Notes

> Renamed from "ModelForge". Package: `llm_parametizer/`. Data dir: `~/.llm_parametizer`
> (auto-migrated from the legacy `~/.modelforge` on first run).

## Where We Are

**App is functionally code-complete.** All core features working. One polish session remains.

Tabs live: **Chat**, **Parameters**, **Tune**, **RAG**. Settings is a modal dialog.

```
LLM-Parametizer/                 (repo folder still named "ModelForge" on disk)
├── main.py                      # python main.py
├── requirements.txt
├── llm_parametizer/
│   ├── core/                    # config, models, paths, memory, state,
│   │                            #   library, families, tuning, rag, ipc
│   ├── backend/                 # process, llama_cpp, ollama, controller
│   ├── chat/                    # history, engine, session
│   ├── tune/                    # runner (threaded sweep)
│   └── ui/                      # theme, widgets, chat_panel, models_panel,
│                                #   tune_panel, rag_panel, settings_*, app
└── tests/                       # test_phase1..6 + test_phase3_ui_smoke
```

## Run / Verify

```powershell
# from C:\Users\kchor\CascadeProjects\ModelForge
python main.py

python tests\test_phase1.py; python tests\test_phase2.py; python tests\test_phase2_streaming.py; python tests\test_phase3_controller.py; python tests\test_phase4.py; python tests\test_phase5.py; python tests\test_phase6.py; python tests\test_phase3_ui_smoke.py
```

## Recently Completed (This Session)

- **IPC bridge** (`core/ipc.py`) — HTTP endpoint at 127.0.0.1:9820 for
  Shield/Tester cross-extension communication. CORS, action allow-list.
- **Duplicate --ctx-size** fixed in profile/command generation.
- **Auto-refresh backend status** when switching to Tune tab.
- **Tune Cancel** — immediate UI reset, late callback guard, partial results shown.
- **RAG workflow** — dialog prompt to switch to Chat after adding docs.
- **RAG layout** — compact docs list (80px), Test Search gets proper space.
- **"Excerpts per question"** — inline explanation added.
- **Backend stop button** — renamed to "■ Stop Model" (gold), distinct from
  chat "■ Cancel" (red). No more confusion.
- **False "Error" after stop** — race condition fixed with `_stopping` flag
  in ProcessManager so intentional stops don't show scary Error pill.
- **Orphan process fix** — hardened stop() + kill_orphans on close + atexit.

## Session 2 — COMPLETE ✅ (UI polish)

1. ✅ **Settings split into two tabs** — "Model & Server" / "Appearance"
   (CTkTabview inside the dialog).
2. ✅ **Settings font fixed** — entry fields now use `fs_button()` (height 30)
   to match their bold labels; no more tiny mismatched inputs.
3. ✅ **RAG headings** — now use `theme.fs_heading()`, consistent with Settings.
4. ✅ **Tune spinner + percentage** — braille spinner + live `NN%` label beside
   the progress bar; start/stop wired into run/progress/cancel/finish/error.
5. ✅ **Home button** — persistent "🏠 Chat" pinned right of the tab bar for
   one-click return from any utility/config tab.
6. ✅ **Verified** — 105/105 tests pass; all UI modules import clean.

**App is feature-complete.** No Session 3 planned.

## Decided to Skip

- **Export/import bundles** — INI files can be emailed; IPC handles config distribution.
- **RAG embeddings** — BM25 is fast and sufficient for technical docs.

## Watch-outs
- Repo **folder** on disk is still `ModelForge` (not renamed to avoid cwd churn).
  Rename later if desired; update no imports (package is `llm_parametizer`).
- `~/.llm_parametizer` holds settings, library, sessions, RAG cache.
- Legacy reference (frozen): `C:\Users\kchor\Desktop\modelforge`.
- LLM Shield code: `C:\Users\kchor\Desktop\LLM Shield\`
- LLM Tester code: `D:\LLM Tester\`

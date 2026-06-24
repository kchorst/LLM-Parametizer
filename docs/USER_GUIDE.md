# LLM Parametizer — User's Guide

This guide walks through the whole app, in the order you'll typically use it.

- [1. Setup & backends](#1-setup--backends)
- [2. The layout](#2-the-layout)
- [3. Chat (USE)](#3-chat-use)
- [4. RAG — your documents (USE)](#4-rag--your-documents-use)
- [5. Parameters (CONFIGURE)](#5-parameters-configure)
- [6. Tune — automated sweeps (CONFIGURE)](#6-tune--automated-sweeps-configure)
- [7. Report — snapshots & history (CONFIGURE)](#7-report--snapshots--history-configure)
- [8. Downloading models](#8-downloading-models)
- [9. Settings](#9-settings)
- [10. Integration with LLM Tester & Sentinel+](#10-integration-with-llm-tester--sentinel)
- [11. Troubleshooting](#11-troubleshooting)

---

## 1. Setup & backends

Run the app:

```bash
python main.py
```

Open **⚙ Settings** (top‑right of the Chat tab) and configure:

- **GGUF models folder** — where your `.gguf` model files live. Downloads also
  land here.
- **llama.cpp Bin directory** — the folder containing `llama-server`
  (and optionally `llama-cli`).
- *(Optional)* **Ollama host/port** — if you use Ollama as a secondary backend.

Backends supported:

| Engine | Role | Notes |
|---|---|---|
| **llama.cpp (server)** | Primary | Has a port; this is the chat target. Started/stopped from Chat. |
| **llama.cpp (cli)** | One‑shot | No chat server — use the server engine to chat. |
| **Ollama** | Secondary | External; the app does not start it. Pick an installed model. |
| **LM Studio / vLLM / text-generation-webui** | Compatible | Any OpenAI-compatible server works via the llama.cpp (server) mode. |

---

## 2. The layout

The top bar groups tabs by intent:

```
USE   [ Chat ]  [ RAG ]      │      CONFIGURE   [ Tune ]  [ Parameters ]  [ Report ]
```

- **USE** tools (blue) are where you *use* the model.
- **CONFIGURE** tools (purple) are where you *shape* its behavior.

When you change the model or parameters, a hint bar suggests re‑testing across
the tabs.

---

## 3. Chat (USE)

1. Choose an **Engine** and **Model** from the dropdowns (click **↻** to rescan).
2. Click **▶ Start** to launch the backend. The status pill shows
   **Starting… → Running** (or **Stopped**).
3. Type a message and press **Enter** (use **Shift+Enter** for a newline).

Top‑bar buttons:

- **⚙ Settings** — open settings.
- **⬇ Models** — search & download GGUF models (see §8).
- **↻** — rescan the models folder / Ollama.
- **Flush RAM** — release memory.
- **Copy / Notes / Logs / 📁 Data** — copy the transcript, jot notes, view
  backend logs, or open the data folder.

**Metrics bar** (appears after first run): Shows TPS (tokens per second), TTFT (time to first token), and total tokens for the last run. Hidden until data is available.

If RAG is enabled and relevant excerpts are found, the answer is grounded in
them and a **`Sources:`** line lists which document chunks were used.

---

## 4. RAG — your documents (USE)

RAG (Retrieval‑Augmented Generation) lets the model answer using *your*
documents.

1. Go to the **RAG** tab.
2. **Add** `.pdf`, `.txt`, or `.md` files. They're chunked and indexed (BM25).
3. **Enable** RAG.

Now, in **Chat**, each question retrieves the most relevant excerpts and injects
them as context; the response shows its **Sources**. If nothing relevant is
found, the model answers normally and the status line says so.

> RAG also powers grounded scoring in **Tune** — see §6.

---

## 5. Parameters (CONFIGURE)

Manually shape model behavior:

- **Sampling parameters** — temperature, top‑p, top‑k, repeat penalty, etc.
  (sliders with live values).
- **System prompt** — the model's standing instructions.
- **Template (optional)** — a chat template override.
- **Command preview** — the exact backend command your settings produce.

Use the **library** to save/load profiles and export configurations. These
parameters are shared with Chat and used as the **base** for Tune sweeps.

---

## 6. Tune — automated sweeps (CONFIGURE)

Find better settings empirically.

1. Make sure a model is **running** (start it in Chat). The banner shows
   **Running** (green) or **Not Ready** (amber).
2. **Pick parameters to sweep** and edit their candidate values.
3. Enter **test prompts** (one per line).
4. Choose a **Goal** (balanced / accuracy / creative / concise) and an optional
   **Required keyword** (a run only scores well if the answer contains it).
5. *(Optional)* Toggle **Use RAG context** to score configs on
   **document‑grounded** answers (requires documents added + RAG enabled).
6. Click **▶ Run Test**. Watch the progress bar; **■ Cancel** stops it.

**Scoring (0–100):** starts at 90 for a clean answer, then subtracts for
repetition, refusals, and too‑short output, adjusts for the keyword, and adds a
small speed bonus. Results are ranked best‑first.

Select a result to see the **actual answer** and the metric breakdown behind its
score. Then:

- **Apply Best** — load the winning parameters into the active config.
- **Save Best as Profile…** — save them to your library.

> The context window (`num_ctx`) is held fixed during a sweep because changing it
> requires restarting the server.

---

## 7. Report — snapshots & history (CONFIGURE)

Track your tuning experiments and export configurations.

1. Go to the **Report** tab.
2. **Capture Snapshot** saves the current configuration (model, parameters, system prompt) to the snapshot list.
3. Snapshots are displayed as single-line entries with timestamp and model name.
4. Click a snapshot to view its full configuration details.
5. **Export** copies the selected snapshot as a portable profile (INI format) for sharing or backup.
6. **Delete** removes a snapshot from the list.

The action bar at the bottom provides Snapshot, Export, and Delete controls. The status line shows the total number of snapshots.

---

## 8. Downloading models

Click **⬇ Models** in Chat to open the downloader:

1. **Search** Hugging Face for GGUF models (e.g. *"qwen2.5 instruct"*,
   *"llama 3.1 8b"*).
2. Select a model to list its **GGUF files** with sizes.
3. Click **Download** on the quantization you want (e.g. `Q4_K_M`). A progress
   bar tracks the download; **Cancel** aborts it.

Files download into your **GGUF models folder**. When finished, the Chat model
dropdown refreshes automatically.

> Choosing a quant: smaller (e.g. `Q4_K_M`) = less RAM/VRAM, slightly lower
> quality; larger (e.g. `Q6_K`, `Q8_0`) = more memory, higher fidelity.

---

## 9. Settings

- **Models** — GGUF models folder.
- **llama.cpp** — Bin directory, host/port, GPU layers, threads, flash
  attention, continuous batching.
- **Ollama (secondary)** — host/port.
- **Appearance** — tab font size (and other font/color options).

Click **Save** to apply; **Verify** checks your paths.

---

## 10. Integration with LLM Tester & Sentinel+

LLM Parametizer integrates with two companion apps for a complete local AI workflow:

### LLM Tester
A Chrome extension for benchmarking local LLMs with standardized prompts and metrics.

**How it works with Parametizer:**
- LLM Tester detects if Parametizer is running via IPC on `127.0.0.1:9820`.
- After each test run, LLM Tester pushes safe metrics (TPS, TTFT, total tokens, model name) to Parametizer.
- Parametizer stores these peer-reported metrics and exposes them via IPC status.
- In LLM Tester's advanced diagnostics, you'll see Parametizer's presence, model, and VRAM info (if detected).

**What you see:**
- In LLM Tester popup → Advanced Diagnostics: Parametizer section appears only when detected.
- In Parametizer: Metrics from LLM Tester runs are stored internally (future UI display planned).

### Sentinel+
A Chrome extension that protects against prompt injection and data exfiltration on AI sites.

**How it works with Parametizer:**
- Sentinel+ detects Parametizer via IPC and reads its status (model, engine, VRAM).
- Sentinel+ displays Parametizer's metrics in its popup integration section.
- No metrics flow from Sentinel+ to Parametizer (one-way detection only).

**What you see:**
- In Sentinel+ popup → Integration section: Parametizer card shows model, engine, and VRAM (if detected).

### Security & Privacy
- All IPC communication is bound to `127.0.0.1:9820` (localhost only).
- IPC actions are explicitly allow-listed: `ping`, `get_config`, `set_config`, `get_status`, `get_model_info`, `export_profile`, `report_metrics`, `get_metrics`.
- No raw prompts, responses, templates, secrets, or local paths are exposed via IPC.
- Metrics shared are sanitized: TPS, TTFT, total tokens, model name, status only.

### Requirements for Integration
- All three apps must be running simultaneously.
- Parametizer must be launched first (it hosts the IPC server).
- LLM Tester and Sentinel+ must be installed as Chrome extensions.
- No additional configuration needed — detection is automatic.

---

## 11. Troubleshooting

### General Issues
- **"Not Ready" in Tune / "Backend not running"** — start the backend in **Chat**
  first; Tune reuses that running model.
- **No models listed** — set the **GGUF models folder** in Settings, then click
  **↻**. For Ollama, make sure it's running.
- **RAG returns nothing** — confirm documents are added *and* RAG is **enabled**,
  and that your question shares keywords with the documents (retrieval is
  lexical/BM25).
- **Download fails** — it's a live network call to Hugging Face; check
  connectivity. Partial files (`.part`) are cleaned up on failure/cancel.
- **Out of memory** — use a smaller quant, reduce GPU layers, or **Flush RAM**.
- **Logs** — the **Logs** button in Chat shows backend output for debugging.

### Integration Issues
- **LLM Tester or Sentinel+ don't detect Parametizer** — ensure Parametizer is running (it hosts the IPC server on port 9820). Check that no firewall is blocking localhost connections.
- **Metrics not appearing in LLM Tester diagnostics** — verify Parametizer is running before opening LLM Tester. Refresh the diagnostics by closing/reopening the popup.
- **Sentinel+ shows no Parametizer info** — same as above: Parametizer must be running first. Check that Sentinel+ has the correct Parametizer extension ID whitelisted.
- **IPC connection timeout** — if apps report timeouts, ensure Parametizer's IPC server started successfully (check Parametizer logs). Restart Parametizer if needed.

### Backend-Specific Issues
- **llama-server fails to start** — verify the binary path in Settings is correct. Check that the model file exists and is a valid GGUF. Run the binary manually to see error messages.
- **Ollama connection fails** — ensure Ollama is running (`ollama serve`). Check the host/port in Settings match Ollama's configuration (default: `127.0.0.1:11434`).
- **LM Studio / other servers not connecting** — verify the server is running and the host/port in Settings match. Ensure the server exposes the OpenAI `/v1/chat/completions` endpoint.

### Performance Issues
- **Slow generation** — increase GPU layers (if you have a GPU), enable flash attention (if supported), or reduce context size.
- **High memory usage** — use a smaller quantization (Q4_K_M instead of Q6_K/Q8_0), reduce GPU layers, or decrease context window.
- **Metrics bar not showing** — this is normal before the first run. Complete at least one chat generation for metrics to appear.

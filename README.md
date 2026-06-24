# LLM Parametizer

A desktop GUI for **running, tuning, and evaluating local LLMs**. Point it at a
local `llama.cpp` server (or Ollama), chat with a model, sweep sampling
parameters to find the best configuration, ground answers in your own documents
(RAG), and download new GGUF models — all offline, no API keys.

**Integrates with LLM Tester and Sentinel+** for unified metrics and protection across your local AI workflow.

> Part of the **ModelForge** project. App version 0.1.0.

---

## Features

- **Chat** with a local model (streaming), with one-click backend start/stop and real-time metrics (TPS, TTFT).
- **Tune** — automated parameter sweeps (temperature, top‑p, top‑k, repeat
  penalty) scored on a transparent 0–100 metric, with goal presets and an
  optional required keyword. Apply or save the best config as a profile.
- **Parameters** — manual editor for sampling params, system prompt, and chat
  template, with a live command preview and a profile library.
- **RAG** — index your `.pdf`/`.txt`/`.md` documents (offline BM25 retrieval)
  and use them as grounding context in Chat **and** during Tune. Chat shows the
  **Sources** used for each answer.
- **Model download** — search Hugging Face for GGUF models and download a chosen
  quantization straight into your models folder.
- **Report** — capture configuration snapshots, export profiles, and track tuning history.
- **Integration** — share metrics with LLM Tester and Sentinel+ via secure local IPC; detect peer apps automatically.
- Dark, modern UI with configurable fonts/colors.

The tabs are grouped by intent: **USE** (Chat, RAG) and **CONFIGURE**
(Tune, Parameters, Report).

---

## Requirements

- **Python 3.10+** (Windows/macOS/Linux)
- Python packages — see [`requirements.txt`](requirements.txt):
  `customtkinter`, `requests`, `psutil`, `pypdf`, `pdfplumber`, `rank-bm25`,
  `diskcache`, `pyyaml`
- A **backend** (at least one):
  - **llama.cpp** binaries (`llama-server`, optionally `llama-cli`) — set the
    Bin directory in **Settings**.
  - **Ollama** (optional, secondary) — running locally.
  - **LM Studio, vLLM, text-generation-webui** — any OpenAI-compatible server.

## Install

```bash
# from the project root (ModelForge/)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

On first launch, open **⚙ Settings** and set:
- **GGUF models folder** — where your `.gguf` files live (and where downloads land).
- **llama.cpp Bin directory** — folder containing `llama-server` / `llama-cli` (if using llama.cpp).

Then pick a model, click **▶ Start**, and chat. See the
[User's Guide](docs/USER_GUIDE.md) for a full walkthrough including integration with LLM Tester and Sentinel+.

---

## Project layout

```
ModelForge/
├─ main.py                     # entry point: python main.py
├─ requirements.txt
└─ llm_parametizer/
   ├─ core/                    # pure logic: config, models, tuning, rag, downloader, state
   ├─ backend/                 # backend process controller
   ├─ chat/                    # chat engine (streaming)
   ├─ tune/                    # tune runner (executes sweeps)
   └─ ui/                      # CustomTkinter UI: app, chat/tune/models/rag panels, dialogs
```

## Notes

- **Offline by design.** The only network calls are the optional **model
  download** (Hugging Face) and your local backend.
- **num_ctx is fixed during a sweep** — changing it requires restarting the
  server, so Tune holds it constant.
- RAG retrieval is **BM25 (lexical)**; semantic embeddings are planned.
- **Security** — IPC bridge binds only to `127.0.0.1:9820` with an explicit action allow-list. No raw prompts, responses, or secrets are exposed.
- **Integration** — LLM Tester and Sentinel+ can detect Parametizer and share metrics via secure local IPC. All three apps must be running for full integration.

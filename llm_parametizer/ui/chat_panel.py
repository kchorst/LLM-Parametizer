"""
ChatPanel — the working chat tab.

Wires together:
  - BackendController (engine, model, start/stop, status)
  - ChatSession (history + streaming)
  - StatusPill + MemoryMonitor

All backend/session callbacks run on daemon threads; every UI mutation is
marshalled onto the Tk thread via self.after(0, ...).
"""

from __future__ import annotations

import os
import sys
import customtkinter as ctk
from tkinter import messagebox, Menu

from . import theme
from .widgets import StatusPill, MemoryMonitor, MetricsBar
from ..core.config import settings, SESSIONS_DIR, APP_DIR
from ..core import paths, memory
from ..core.metrics import SessionMetrics
from ..core.models import default_params
from ..backend.controller import (
    BackendController, ENGINES, ENGINE_OLLAMA, ENGINE_LLAMA_CLI,
)
from ..backend import ollama as ollama_client
from ..chat.session import ChatSession
from ..chat.engine import Backend, ChatStats


class ChatPanel(ctk.CTkFrame):
    def __init__(self, master, active=None, rag_store=None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)

        # Shared active config (params + prompts edited in the Models tab).
        from ..core.state import ActiveConfig
        self.active = active if active is not None else ActiveConfig()

        # Shared RAG store (documents indexed in the RAG tab). May be None.
        self.rag_store = rag_store
        # Source labels for the chunks used on the in-flight message (citations).
        self._pending_citations: list[str] = []

        # Session performance metrics (TTFT/TPS/tokens across messages).
        self.session_metrics = SessionMetrics()

        # State
        self._gguf_entries = []           # list[GGUFEntry]
        self._streaming_active = False
        self._assistant_open = False      # whether an assistant block is being streamed
        self._last_user_prompt = ""       # for regenerate
        self._context_used_tokens = 0
        self._context_limit_tokens = settings.get_int("n_ctx", 4096)

        # Vars
        self.engine_var = ctk.StringVar(value=settings.get_str("default_backend", ENGINES[0]))
        self.model_var = ctk.StringVar(value="(no model)")
        self.status_msg = ctk.StringVar(value="")

        # Set by the app to open the Settings dialog.
        self.on_open_settings = lambda: None
        # Set by the app; fired when the user changes engine or model.
        self.on_model_changed = lambda: None

        # Backend controller (callbacks marshalled to UI thread)
        self.controller = BackendController(
            settings,
            on_status=lambda s: self.after(0, self._on_backend_status, s),
            on_log=lambda l: self.after(0, self._on_backend_log, l),
            on_ready=lambda: self.after(0, self._on_backend_ready),
        )

        # Chat session
        self.session = ChatSession(
            on_start=lambda: self.after(0, self._on_chat_start),
            on_token=lambda t: self.after(0, self._on_chat_token, t),
            on_done=lambda f, s: self.after(0, self._on_chat_done, f, s),
            on_error=lambda m: self.after(0, self._on_chat_error, m),
        )

        self._build()
        self.after(100, self._post_init)

    # ── Layout ──────────────────────────────────────────────────────────────────

    def _build(self):
        # Top control bar
        bar = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0)
        bar.pack(fill="x", padx=0, pady=0)

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=8)

        ctk.CTkLabel(inner, text="Engine",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        self.engine_combo = ctk.CTkComboBox(
            inner, variable=self.engine_var, values=list(ENGINES),
            command=self._on_engine_pick, width=170, height=30,
            font=ctk.CTkFont(size=theme.fs_small()),
        )
        self.engine_combo.pack(side="left", padx=(8, 16))

        ctk.CTkLabel(inner, text="Model",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        self.model_combo = ctk.CTkComboBox(
            inner, variable=self.model_var, values=["(no model)"],
            command=self._on_model_pick,
            width=280, height=30, font=ctk.CTkFont(size=theme.fs_small()),
        )
        self.model_combo.pack(side="left", padx=(8, 16))

        self.start_btn = ctk.CTkButton(
            inner, text="▶ Start", width=90, height=30,
            fg_color=theme.SUCCESS, hover_color="#1F8B4C",
            command=self._on_start_stop,
            font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.status_pill = StatusPill(inner, label="Stopped")
        self.status_pill.pack(side="left", padx=(14, 0))

        # Right side: settings + refresh + flush
        ctk.CTkButton(inner, text="⚙ Settings", width=96, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=lambda: self.on_open_settings(),
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=(8, 0))
        ctk.CTkButton(inner, text="⬇ Models", width=92, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._open_model_downloader,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=(8, 0))
        ctk.CTkButton(inner, text="↻", width=32, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._refresh_models,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right")
        ctk.CTkButton(inner, text="Flush RAM", width=90, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._flush_memory,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=(0, 8))
        ctk.CTkButton(inner, text="Copy", width=64, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._copy_transcript,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=(0, 8))
        ctk.CTkButton(inner, text="Notes", width=64, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._open_notes,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=(0, 8))
        ctk.CTkButton(inner, text="Logs", width=64, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._open_log_viewer,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=(0, 8))
        ctk.CTkButton(inner, text="📁 Data", width=74, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._open_data_folder,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=(0, 8))

        # Chat history
        self.history = ctk.CTkTextbox(
            self, fg_color=theme.SURFACE,
            font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
            text_color=theme.TEXT(), wrap="word",
        )
        self.history.pack(fill="both", expand=True, padx=12, pady=(10, 6))
        self.history.configure(state="disabled")
        self._build_context_menu()

        # Status line
        self.status_label = ctk.CTkLabel(
            self, textvariable=self.status_msg,
            font=ctk.CTkFont(size=theme.fs_small()), text_color=theme.MUTED(),
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=12, pady=(0, 4))

        # Live performance metrics (last message) + session averages.
        # Hidden entirely until a run produces real data — no blank metrics.
        self.metrics_row = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_bar = MetricsBar(self.metrics_row, title="Last:")
        self.metrics_bar.pack(side="left")
        self.session_lbl = ctk.CTkLabel(
            self.metrics_row, text="", anchor="e",
            font=ctk.CTkFont(size=theme.fs_small()), text_color=theme.MUTED())
        self.session_lbl.pack(side="right", padx=(8, 0))

        # Context usage meter
        self.ctx_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctx_frame = self.ctx_frame
        ctx_frame.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(ctx_frame, text="Context:", width=60, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(side="left")
        self.ctx_bar = ctk.CTkProgressBar(ctx_frame, width=200, height=6,
                                          progress_color=theme.ACCENT())
        self.ctx_bar.pack(side="left", padx=(6, 8))
        self.ctx_bar.set(0)
        self.ctx_label = ctk.CTkLabel(ctx_frame, text="0 / 4096",
                                       font=ctk.CTkFont(size=theme.fs_small()),
                                       text_color=theme.MUTED())
        self.ctx_label.pack(side="left")

        # Input row
        input_row = ctk.CTkFrame(self, fg_color="transparent")
        input_row.pack(fill="x", padx=12, pady=(0, 8))

        self.input = ctk.CTkTextbox(
            input_row, height=70, fg_color=theme.SURFACE2,
            font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
            text_color=theme.TEXT(), wrap="word",
        )
        self.input.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.input.bind("<Return>", self._on_return)
        self.input.bind("<Shift-Return>", self._on_shift_return)

        right = ctk.CTkFrame(input_row, fg_color="transparent")
        right.pack(side="right", fill="y")

        # Action buttons: Send (normal), Stop (streaming), Regenerate (after stop)
        self.send_btn = ctk.CTkButton(
            right, text="Send", width=90, height=34,
            fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(),
            command=self._on_send,
            font=ctk.CTkFont(size=theme.fs_base(), weight="bold"),
        )
        self.send_btn.pack(side="top")

        self.stop_btn = ctk.CTkButton(
            right, text="■ Cancel", width=90, height=34,
            fg_color=theme.DANGER, hover_color="#A03228",
            command=self._on_stop_generation,
            font=ctk.CTkFont(size=theme.fs_base(), weight="bold"),
        )
        # hidden initially
        self.stop_btn.pack_forget()

        self.regen_btn = ctk.CTkButton(
            right, text="↻ Regen", width=90, height=34,
            fg_color=theme.WARNING, hover_color="#C9A227",
            text_color="#10141C",
            command=self._on_regenerate,
            font=ctk.CTkFont(size=theme.fs_base(), weight="bold"),
        )
        # hidden initially
        self.regen_btn.pack_forget()

        ctk.CTkButton(
            right, text="Clear", width=90, height=28,
            fg_color=theme.SURFACE2, hover_color=theme.BORDER,
            command=self._clear_chat,
            font=ctk.CTkFont(size=theme.fs_small()),
        ).pack(side="top", pady=(4, 0))

        # Bottom: memory monitor
        self.mem = MemoryMonitor(self)
        self.mem.pack(fill="x", padx=12, pady=(0, 8))

    # ── Init ──────────────────────────────────────────────────────────────────

    def _post_init(self):
        self.controller.kill_orphans()
        self._on_engine_change()
        self._refresh_models()

    # ── Engine / model ─────────────────────────────────────────────────────────

    def _on_engine_pick(self, val=None):
        # User explicitly changed the engine via the combo.
        self._on_engine_change(val)
        try:
            self.on_model_changed()
        except Exception:
            pass

    def _on_model_pick(self, _val=None):
        # User explicitly selected a different model.
        try:
            self.on_model_changed()
        except Exception:
            pass

    def _on_engine_change(self, _val=None):
        engine = self.engine_var.get()
        self.controller.set_engine(engine)
        settings["default_backend"] = engine
        settings.save()
        self._refresh_models()
        # Start button is irrelevant for Ollama (external)
        if engine == ENGINE_OLLAMA:
            self.start_btn.configure(state="disabled")
            self.status_pill.set("External", theme.INFO)
        else:
            self.start_btn.configure(state="normal")
            if not self.controller.is_running:
                self.status_pill.set_state("stopped")

    def _refresh_models(self):
        engine = self.engine_var.get()
        if engine == ENGINE_OLLAMA:
            host = settings.get_str("ollama_host", "127.0.0.1")
            port = settings.get_int("ollama_port", 11434)
            names = ollama_client.list_models(host, port)
            if names:
                self.model_combo.configure(values=names)
                if self.model_var.get() not in names:
                    self.model_var.set(names[0])
            else:
                self.model_combo.configure(values=["(Ollama not running)"])
                self.model_var.set("(Ollama not running)")
            return

        # llama.cpp: scan GGUF folder
        models_dir = settings.get_str("gguf_models_dir")
        self._gguf_entries = paths.scan_gguf_models(models_dir)
        if not models_dir:
            self.model_combo.configure(values=["(set models folder in Settings)"])
            self.model_var.set("(set models folder in Settings)")
        elif not self._gguf_entries:
            self.model_combo.configure(values=["(no .gguf files found)"])
            self.model_var.set("(no .gguf files found)")
        else:
            displays = [e.display_name for e in self._gguf_entries]
            self.model_combo.configure(values=displays)
            if self.model_var.get() not in displays:
                self.model_var.set(displays[0])

    def _open_model_downloader(self):
        from .model_download_dialog import ModelDownloadDialog
        existing = getattr(self, "_download_dialog", None)
        if existing is not None and existing.winfo_exists():
            existing.focus()
            return
        # Refresh the model dropdown whenever a download finishes.
        self._download_dialog = ModelDownloadDialog(
            self.winfo_toplevel(), on_downloaded=self._refresh_models)

    def _selected_gguf_path(self) -> str:
        display = self.model_var.get()
        for e in self._gguf_entries:
            if e.display_name == display:
                return e.full_path
        return ""

    # ── Start / Stop ─────────────────────────────────────────────────────────

    def _on_start_stop(self):
        if self.controller.is_running:
            self.status_msg.set("Stopping…")
            self.controller.stop()
            return

        engine = self.engine_var.get()
        if engine != ENGINE_OLLAMA:
            gguf = self._selected_gguf_path()
            if not gguf:
                messagebox.showwarning("No model",
                                       "Select a .gguf model first (set the models folder in Settings).",
                                       parent=self)
                return
            self.controller.set_gguf_path(gguf)
            self.controller.set_params(self._current_params())
            self.controller.set_system_prompt(self.active.system_prompt)

        ok, msg = self.controller.start()
        self.status_msg.set(msg)
        if not ok:
            messagebox.showerror("Cannot start", msg, parent=self)

    def _current_params(self) -> dict:
        # Phase 4: parameters come from the shared ActiveConfig (Models tab).
        return dict(self.active.params)

    # ── Backend callbacks (UI thread) ──────────────────────────────────────────

    def _on_backend_status(self, status: str):
        self.status_pill.set_state(status)
        if status == "ready":
            self.start_btn.configure(text="■ Stop Model", fg_color=theme.WARNING, hover_color="#C9A227",
                                    text_color="#10141C")
            self.status_msg.set(f"Backend ready — {self.engine_var.get()}")
        elif status in ("stopped", "error"):
            self.start_btn.configure(text="▶ Start", fg_color=theme.SUCCESS, hover_color="#1F8B4C",
                                    text_color="#DCE4EE")
            if status == "error":
                self.status_msg.set("Backend exited with an error — see log.")
            memory.flush_memory()

    def _on_backend_log(self, line: str):
        # Phase 3: surface to status line; full log viewer in Phase 4.
        if "error" in line.lower() or "failed" in line.lower():
            self.status_msg.set(line[:120])

    def _on_backend_ready(self):
        pass  # handled in status

    # ── Send / streaming ─────────────────────────────────────────────────────

    def _on_return(self, _e):
        self._on_send()
        return "break"

    def _on_shift_return(self, _e):
        return None  # allow newline

    def current_chat_target(self):
        """
        Resolve the active chat target for reuse (e.g. AutoTune).
        Returns (backend, host, port, model, ok, reason).
        """
        engine = self.engine_var.get()
        backend, host, port, _ = self.controller.chat_target()
        model = ""
        if engine == ENGINE_OLLAMA:
            model = self.model_var.get()
            if model.startswith("("):
                return backend, host, port, "", False, "Select a valid Ollama model."
        else:
            if engine == ENGINE_LLAMA_CLI:
                return backend, host, port, "", False, "llama-cli has no server — use llama.cpp (server)."
            if not self.controller.chat_reachable():
                return backend, host, port, "", False, "Start the llama-server backend first."
        return backend, host, port, model, True, ""

    def _on_send(self, prompt: str = ""):
        if self.session.is_busy:
            return
        if not prompt:
            prompt = self.input.get("1.0", "end").strip()
        if not prompt:
            return

        engine = self.engine_var.get()
        backend, host, port, _ = self.controller.chat_target()
        model = ""
        if engine == ENGINE_OLLAMA:
            model = self.model_var.get()
            if model.startswith("("):
                messagebox.showwarning("No model", "Select a valid Ollama model.", parent=self)
                return
        else:
            if engine == ENGINE_LLAMA_CLI:
                messagebox.showinfo("CLI mode",
                                    "llama-cli has no chat server. Use 'llama.cpp (server)' to chat.",
                                    parent=self)
                return
            if not self.controller.chat_reachable():
                messagebox.showwarning("Backend not running",
                                       "Start the llama-server backend before chatting.",
                                       parent=self)
                return

        # Store for regenerate
        self._last_user_prompt = prompt
        self._context_limit_tokens = settings.get_int("n_ctx", 4096)

        # Clear any previous cancel state
        self.session._cancel.clear()

        if not self.input.get("1.0", "end").strip() == "":
            self.input.delete("1.0", "end")
        self._append_block("You", prompt)

        # Toggle buttons: hide Send, show Stop
        self.send_btn.pack_forget()
        self.regen_btn.pack_forget()
        self.stop_btn.pack(side="top")

        # Reset context tracking
        self._context_used_tokens = 0
        self._update_context_meter()

        started = self.session.send(
            prompt, backend,
            host=host, port=port, model=model,
            system_prompt=self._system_prompt_for(prompt),
            num_ctx=self._current_params().get("num_ctx", self._context_limit_tokens),
            options=self._current_params(),
        )
        if not started:
            self._show_send_button()

    def _on_stop_generation(self):
        """Cancel the current generation and show Regenerate button."""
        self.session.cancel()
        self.status_msg.set("Stopping…")

    def _on_regenerate(self):
        """Regenerate the last user message."""
        if not self._last_user_prompt:
            self.status_msg.set("Nothing to regenerate.")
            return
        # Clear cancel flag from previous stop
        self.session._cancel.clear()
        # Reset state and resend
        self._show_send_button()
        self._on_send(prompt=self._last_user_prompt)

    def _show_send_button(self):
        """Show Send button, hide Stop and Regen."""
        self.stop_btn.pack_forget()
        self.regen_btn.pack_forget()
        self.send_btn.pack(side="top")
        self.send_btn.configure(state="normal")

    def _show_regen_button(self):
        """Show Regenerate button, hide Stop and Send."""
        self.stop_btn.pack_forget()
        self.send_btn.pack_forget()
        self.regen_btn.pack(side="top")

    def refresh_context_limit(self):
        """Re-read n_ctx from settings and update the meter display."""
        self._context_limit_tokens = settings.get_int("n_ctx", 4096)
        self._update_context_meter(self._context_used_tokens)

    def _update_context_meter(self, used: int = 0):
        """Update the context usage progress bar."""
        self._context_used_tokens = used
        limit = max(1, self._context_limit_tokens)
        ratio = min(1.0, used / limit)
        self.ctx_bar.set(ratio)
        self.ctx_label.configure(text=f"{used} / {limit}")
        # Color shift when nearing limit
        if ratio > 0.9:
            self.ctx_bar.configure(progress_color=theme.DANGER)
        elif ratio > 0.7:
            self.ctx_bar.configure(progress_color=theme.WARNING)
        else:
            self.ctx_bar.configure(progress_color=theme.ACCENT())

    def _system_prompt_for(self, prompt: str) -> str:
        """Active system prompt, augmented with RAG context when enabled."""
        base = self.active.system_prompt
        self._pending_citations = []
        store = self.rag_store
        if store and store.enabled and store.has_docs:
            context, chunks = store.context_for(prompt)
            if context:
                self._pending_citations = [c.label for c in chunks]
                self.status_msg.set(f"Using {len(chunks)} document excerpt(s) as context.")
                return f"{base}\n\n{context}" if base else context
            self.status_msg.set("No relevant document excerpts found — answering without them.")
        return base

    def _on_chat_start(self):
        self.status_msg.set("Thinking…")
        self._open_assistant_block()
        # Ensure Stop button is showing
        self.send_btn.pack_forget()
        self.regen_btn.pack_forget()
        self.stop_btn.pack(side="top")

    def _on_chat_token(self, text: str):
        self._append_assistant_text(text)
        # Rough token estimation: ~4 chars per token for English text
        estimated_tokens = max(1, len(text) // 4)
        self._context_used_tokens += estimated_tokens
        self._update_context_meter(self._context_used_tokens)

    def _on_chat_done(self, full: str, stats: ChatStats):
        self._close_assistant_block()
        if self._pending_citations:
            self._append_block("Sources", " · ".join(self._pending_citations))
            self._pending_citations = []

        # Update context meter with actual token count if available
        if stats and hasattr(stats, 'tokens') and stats.tokens:
            self._update_context_meter(stats.tokens)

        # Determine if stopped early vs completed normally
        was_cancelled = self.session._cancel.is_set()
        if was_cancelled:
            self.status_msg.set("Stopped — click ↻ Regen to retry")
            self._show_regen_button()
        else:
            self._show_send_button()
            if stats and stats.tokens_per_second:
                self.status_msg.set(f"Done — {stats.tokens_per_second} t/s, {stats.elapsed}s")
            elif stats:
                self.status_msg.set(f"Done in {stats.elapsed}s")
            else:
                self.status_msg.set("Done")
            self._update_metrics(stats)

    def _on_chat_error(self, msg: str):
        self._close_assistant_block()
        self._pending_citations = []
        self._show_send_button()
        self.status_msg.set(f"Error: {msg[:120]}")
        self._append_block("System", f"⚠ {msg}")

    def _update_metrics(self, stats: ChatStats):
        """Update the live metrics chips and the running session averages."""
        if not stats:
            return
        self.metrics_bar.set_metrics(
            ttft=stats.ttft, tps=stats.tokens_per_second,
            tokens=stats.tokens or None, elapsed=stats.elapsed or None)
        self.session_metrics.add(stats)
        sm = self.session_metrics
        if sm.count:
            parts = [f"session: {sm.count} msg"]
            if sm.avg_tps is not None:
                parts.append(f"avg {sm.avg_tps} tok/s")
            if sm.avg_ttft is not None:
                parts.append(f"avg TTFT {sm.avg_ttft}s")
            parts.append(f"{sm.total_tokens} tok total")
            self.session_lbl.configure(text="  ·  ".join(parts))
        # Reveal the metrics row only once we actually have data to show.
        if (self.metrics_bar.has_data or sm.count) and not self.metrics_row.winfo_ismapped():
            self.metrics_row.pack(fill="x", padx=12, pady=(0, 4),
                                  before=self.ctx_frame)

    # ── History rendering ─────────────────────────────────────────────────────

    def _append_block(self, label: str, content: str):
        self.history.configure(state="normal")
        self.history.insert("end", f"{label}:\n{content.strip()}\n\n")
        self.history.see("end")
        self.history.configure(state="disabled")

    def _open_assistant_block(self):
        self.history.configure(state="normal")
        self.history.insert("end", "Assistant:\n")
        self.history.see("end")
        self.history.configure(state="disabled")
        self._assistant_open = True

    def _append_assistant_text(self, text: str):
        if not self._assistant_open:
            self._open_assistant_block()
        self.history.configure(state="normal")
        self.history.insert("end", text)
        self.history.see("end")
        self.history.configure(state="disabled")

    def _close_assistant_block(self):
        if self._assistant_open:
            self.history.configure(state="normal")
            self.history.insert("end", "\n\n")
            self.history.see("end")
            self.history.configure(state="disabled")
            self._assistant_open = False

    def _clear_chat(self):
        self.session.clear()
        self.history.configure(state="normal")
        self.history.delete("1.0", "end")
        self.history.configure(state="disabled")
        self.status_msg.set("")
        self._last_user_prompt = ""
        self._context_used_tokens = 0
        self._update_context_meter(0)
        self.session_metrics.reset()
        self.metrics_bar.clear()
        self.session_lbl.configure(text="")
        self.metrics_row.pack_forget()
        self._show_send_button()

    # ── Copy / context menu (Phase 7) ────────────────────────────────────────

    def _build_context_menu(self):
        menu = Menu(self, tearoff=0)
        menu.add_command(label="Copy selection", command=self._copy_selection)
        menu.add_command(label="Copy full transcript", command=self._copy_transcript)
        menu.add_separator()
        menu.add_command(label="Clear chat", command=self._clear_chat)
        self._ctx_menu = menu
        # Right-click (and macOS Control-click) on the history widget.
        inner = getattr(self.history, "_textbox", self.history)
        for seq in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
            inner.bind(seq, self._show_context_menu)

    def _show_context_menu(self, event):
        try:
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._ctx_menu.grab_release()
        return "break"

    def _copy_selection(self):
        inner = getattr(self.history, "_textbox", self.history)
        try:
            text = inner.get("sel.first", "sel.last")
        except Exception:
            text = ""
        if text:
            self._to_clipboard(text)
            self.status_msg.set("Copied selection to clipboard.")
        else:
            self._copy_transcript()

    def _copy_transcript(self):
        text = self.session.history.transcript().strip()
        if not text:
            self.status_msg.set("Nothing to copy yet.")
            return
        self._to_clipboard(text)
        self.status_msg.set("Copied full transcript to clipboard.")

    def _to_clipboard(self, text: str):
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            pass

    # ── Session notes (Phase 7) ──────────────────────────────────────────────

    def _notes_path(self):
        return SESSIONS_DIR / "notes.txt"

    def _open_notes(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Session Notes")
        dlg.geometry("560x420")
        dlg.transient(self.winfo_toplevel())
        dlg.configure(fg_color=theme.SURFACE)

        ctk.CTkLabel(dlg, text="Session notes",
                     font=ctk.CTkFont(size=theme.fs_base() + 2, weight="bold"),
                     text_color=theme.TEXT()).pack(anchor="w", padx=18, pady=(16, 2))
        path = self._notes_path()
        ctk.CTkLabel(dlg, text=f"Saved to: {path}", anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(anchor="w", padx=18, pady=(0, 8))

        box = ctk.CTkTextbox(dlg, fg_color=theme.SURFACE2,
                             font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
                             text_color=theme.TEXT(), wrap="word")
        box.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        try:
            if path.is_file():
                box.insert("1.0", path.read_text(encoding="utf-8"))
        except Exception:
            pass

        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(fill="x", padx=18, pady=(0, 16))

        def _save():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(box.get("1.0", "end").rstrip() + "\n", encoding="utf-8")
                self.status_msg.set(f"Notes saved to {path}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e), parent=dlg)
                return
            dlg.destroy()

        ctk.CTkButton(btns, text="Save", width=100, height=34,
                      fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(), text_color="#10141C",
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      command=_save).pack(side="right")
        ctk.CTkButton(btns, text="Close", width=90, height=34,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=dlg.destroy).pack(side="right", padx=(0, 8))
        box.focus_set()

    # ── Backend log viewer (Phase 7) ─────────────────────────────────────────

    def _open_log_viewer(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Backend Log")
        dlg.geometry("760x520")
        dlg.transient(self.winfo_toplevel())
        dlg.configure(fg_color=theme.SURFACE)

        header = ctk.CTkFrame(dlg, fg_color=theme.SURFACE2, corner_radius=8)
        header.pack(fill="x", padx=12, pady=(12, 6))
        status_var = ctk.StringVar(value="")
        ctk.CTkLabel(header, textvariable=status_var, anchor="w",
                     font=ctk.CTkFont(family="Consolas", size=theme.fs_small()),
                     text_color=theme.TEXT()).pack(side="left", padx=10, pady=6)
        autoscroll = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(header, text="Auto-scroll", variable=autoscroll,
                      font=ctk.CTkFont(size=theme.fs_small()),
                      progress_color=theme.SUCCESS).pack(side="right", padx=10)

        box = ctk.CTkTextbox(dlg, fg_color="#0E1116",
                             font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
                             text_color="#C8D3E0", wrap="none")
        box.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        box.configure(state="disabled")

        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(fill="x", padx=12, pady=(0, 12))

        def _copy():
            self._to_clipboard(self.controller.log_text())
        def _clear():
            self.controller.clear_log()

        ctk.CTkButton(btns, text="Copy log", width=90, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER, command=_copy,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="left")
        ctk.CTkButton(btns, text="Clear", width=80, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER, command=_clear,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(8, 0))
        ctk.CTkButton(btns, text="Close", width=80, height=30,
                      fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(), text_color="#10141C",
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      command=dlg.destroy).pack(side="right")

        state = {"count": -1}

        def _refresh():
            if not dlg.winfo_exists():
                return
            # Header: engine, PID, connection health.
            pid = self.controller.pid
            running = self.controller.is_running
            reachable = self.controller.chat_reachable()
            dot = "●"
            health = (f"{dot} reachable" if reachable
                      else (f"{dot} starting…" if running else f"{dot} not running"))
            status_var.set(f"Engine: {self.engine_var.get()}   "
                           f"PID: {pid if pid else '—'}   "
                           f"Lines: {self.controller.log_line_count()}   {health}")
            # Body: only rewrite when new lines arrived (avoids flicker/cost).
            count = self.controller.log_line_count()
            if count != state["count"]:
                state["count"] = count
                box.configure(state="normal")
                box.delete("1.0", "end")
                box.insert("1.0", self.controller.log_text())
                if autoscroll.get():
                    box.see("end")
                box.configure(state="disabled")
            dlg.after(800, _refresh)

        _refresh()

    def _open_data_folder(self):
        path = str(APP_DIR)
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                import subprocess
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, path])
            self.status_msg.set(f"Opened data folder: {path}")
        except Exception as e:
            messagebox.showinfo("Data folder", f"{path}\n\n({e})", parent=self)

    # ── Memory ──────────────────────────────────────────────────────────────────

    def _flush_memory(self):
        n = memory.flush_memory()
        self.mem.refresh()
        self.status_msg.set(f"Flushed memory (gc collected {n} objects).")

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        try:
            self.controller.stop_blocking()
        except Exception:
            pass

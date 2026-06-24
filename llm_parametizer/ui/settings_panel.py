"""
SettingsPanel — paths & backend configuration (Phase 3 quick-start subset).

Lets the user set the essentials needed for chat to work:
  - GGUF models folder
  - llama.cpp bin directory (or explicit binary paths)
  - host / port
  - Ollama host / port
  - GPU layers, threads, flash-attn, context defaults

Saves directly to the Settings singleton. A callback notifies the app so the
Chat tab can refresh its model list and controller.
"""

from __future__ import annotations

import customtkinter as ctk
from tkinter import filedialog

from . import theme
from ..core.config import settings
from ..core import paths


class SettingsPanel(ctk.CTkFrame):
    def __init__(self, master, on_saved=None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)
        self._on_saved = on_saved or (lambda: None)
        self._vars: dict[str, ctk.StringVar | ctk.BooleanVar] = {}

        # Fixed footer FIRST (packed bottom) so it is never pushed off-screen,
        # then a tabview fills the remaining space above it. Two tabs separate
        # model/server config from UI/appearance config.
        self._footer = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0)
        self._footer.pack(side="bottom", fill="x")

        self._tabs = ctk.CTkTabview(self, fg_color=theme.BG,
                                    segmented_button_selected_color=theme.ACCENT(),
                                    segmented_button_selected_hover_color=theme.ACCENT2())
        self._tabs.pack(side="top", fill="both", expand=True, padx=4, pady=(4, 0))
        model_tab = self._tabs.add("Model & Server")
        ui_tab = self._tabs.add("Appearance")
        self._model_content = ctk.CTkScrollableFrame(model_tab, fg_color=theme.BG)
        self._model_content.pack(fill="both", expand=True)
        self._ui_content = ctk.CTkScrollableFrame(ui_tab, fg_color=theme.BG)
        self._ui_content.pack(fill="both", expand=True)

        self._build()

    # ── Builders ──────────────────────────────────────────────────────────────

    def _section(self, parent, title: str):
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(size=theme.fs_heading(), weight="bold"),
                     text_color=theme.ACCENT()).pack(anchor="w", padx=14, pady=(14, 4))

    def _hint(self, parent, text: str):
        ctk.CTkLabel(parent, text=text, anchor="w", justify="left", wraplength=620,
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(anchor="w", padx=14, pady=(0, 4))

    def _path_row(self, parent, label: str, key: str, is_dir: bool):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=3)
        ctk.CTkLabel(row, text=label, width=160, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.TEXT()).pack(side="left")
        var = ctk.StringVar(value=settings.get_str(key))
        self._vars[key] = var
        ctk.CTkEntry(row, textvariable=var, width=420, height=30,
                     font=ctk.CTkFont(size=theme.fs_button())).pack(
            side="left", padx=(6, 6))

        def browse():
            if is_dir:
                p = filedialog.askdirectory(parent=self)
            else:
                p = filedialog.askopenfilename(parent=self)
            if p:
                var.set(p)

        ctk.CTkButton(row, text="Browse", width=70, height=28,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=browse,
                      font=ctk.CTkFont(size=theme.fs_button())).pack(side="left")

    def _entry_row(self, parent, label: str, key: str, width: int = 120):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=3)
        ctk.CTkLabel(row, text=label, width=160, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.TEXT()).pack(side="left")
        var = ctk.StringVar(value=str(settings.get(key)))
        self._vars[key] = var
        ctk.CTkEntry(row, textvariable=var, width=width, height=30,
                     font=ctk.CTkFont(size=theme.fs_button())).pack(side="left", padx=(6, 0))

    def _switch_row(self, parent, label: str, key: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=3)
        ctk.CTkLabel(row, text=label, width=160, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.TEXT()).pack(side="left")
        var = ctk.BooleanVar(value=settings.get_bool(key))
        self._vars[key] = var
        ctk.CTkSwitch(row, text="", variable=var, width=44).pack(side="left", padx=(6, 0))

    def _build(self):
        m = self._model_content
        self._section(m, "Models")
        self._path_row(m, "GGUF models folder", "gguf_models_dir", is_dir=True)

        self._section(m, "llama.cpp binaries")
        self._hint(m, "Set the Bin directory to your LLAMA folder — the binaries are "
                   "found automatically. The paths below are optional overrides.")
        self._path_row(m, "Bin directory", "llama_bin_dir", is_dir=True)
        self._path_row(m, "llama-server (optional)", "llama_server_path", is_dir=False)
        self._path_row(m, "llama-cli (optional)", "llama_cli_path", is_dir=False)
        self._path_row(m, "llama-swap (optional)", "llama_swap_path", is_dir=False)

        self._section(m, "llama.cpp server")
        self._entry_row(m, "Host", "llama_host")
        self._entry_row(m, "Port", "llama_port")
        self._entry_row(m, "Context window (n_ctx)", "n_ctx")
        self._entry_row(m, "GPU layers (-1=all)", "n_gpu_layers")
        self._entry_row(m, "Threads (-1=auto)", "threads")
        self._switch_row(m, "Flash attention", "flash_attn")
        self._switch_row(m, "Continuous batching", "cont_batching")

        self._section(m, "Ollama (secondary)")
        self._entry_row(m, "Host", "ollama_host")
        self._entry_row(m, "Port", "ollama_port")

        u = self._ui_content
        self._section(u, "Tab bar")
        self._hint(u, "Font size of the main tab buttons (Chat / Tune / Parameters / RAG).")
        self._entry_row(u, "Tab font size", "tab_font_size")

        # Fixed footer: status line + Save/Verify (always visible).
        self._status = ctk.CTkLabel(self._footer, text="", anchor="w", justify="left",
                                    wraplength=720,
                                    font=ctk.CTkFont(size=theme.fs_small()),
                                    text_color=theme.MUTED())
        self._status.pack(fill="x", padx=14, pady=(8, 2))

        bar = ctk.CTkFrame(self._footer, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkButton(bar, text="Save", width=110, height=32,
                      fg_color=theme.SUCCESS, hover_color="#1F8B4C",
                      command=self._save,
                      font=ctk.CTkFont(size=theme.fs_base(), weight="bold")).pack(side="left")
        ctk.CTkButton(bar, text="Verify", width=90, height=32,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._verify,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(8, 0))

    # ── Actions ──────────────────────────────────────────────────────────────

    def _collect(self):
        int_keys = {"llama_port", "ollama_port", "n_ctx", "n_gpu_layers", "threads", "tab_font_size"}
        for key, var in self._vars.items():
            val = var.get()
            if isinstance(var, ctk.BooleanVar):
                settings[key] = bool(val)
            elif key in int_keys:
                try:
                    settings[key] = int(val)
                except (TypeError, ValueError):
                    settings[key] = 0
            else:
                settings[key] = val

    def _save(self):
        self._collect()
        ok = settings.save()
        self._status.configure(
            text="Saved." if ok else "Save failed.",
            text_color=theme.SUCCESS if ok else theme.DANGER,
        )
        self._on_saved()

    def _verify(self):
        self._collect()
        bin_dir = settings.get_str("llama_bin_dir")
        server = paths.resolve_llama_server(settings.get_str("llama_server_path"), bin_dir)
        cli = paths.resolve_llama_cli(settings.get_str("llama_cli_path"), bin_dir)
        swap = paths.resolve_llama_swap(settings.get_str("llama_swap_path"), bin_dir)
        models_dir = settings.get_str("gguf_models_dir")
        ggufs = paths.scan_gguf_models(models_dir)

        def mark(name: str, path: str) -> str:
            return f"{name}: {'OK' if path else 'NOT FOUND'}"

        lines = [
            mark("llama-server", server),
            mark("llama-cli", cli),
            mark("llama-swap", swap),
            (f"GGUF models: {len(ggufs)} found in folder" if models_dir
             else "GGUF folder: not set"),
        ]
        color = theme.SUCCESS if (server and ggufs) else theme.WARNING
        self._status.configure(text="    |    ".join(lines), text_color=color)

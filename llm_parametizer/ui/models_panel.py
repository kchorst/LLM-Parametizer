"""
ModelsPanel — parameter sliders, prompt editors, command preview, and the
profile library.

Edits update the shared ActiveConfig (which the Chat tab reads). Profiles are
saved/loaded as INI via ProfileLibrary; the live command preview and exports
reuse llama_cpp (build_command_str / to_ini / to_modelfile).
"""

from __future__ import annotations

import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog

from . import theme
from ..core.config import settings
from ..core import paths
from ..core.models import PARAMETERS, PARAM_KEYS, default_params
from ..core.state import ActiveConfig
from ..core.library import ProfileLibrary
from ..backend import llama_cpp as lc
from ..backend.controller import ENGINE_LLAMA_CLI


def _backend_kwargs() -> dict:
    """Backend fields for ModelConfig, pulled from settings + resolved paths."""
    s = settings
    bin_dir = s.get_str("llama_bin_dir")
    mode = "cli" if s.get_str("default_backend") == ENGINE_LLAMA_CLI else "server"
    return dict(
        llama_mode=mode,
        llama_server_path=paths.resolve_llama_server(s.get_str("llama_server_path"), bin_dir),
        llama_cli_path=paths.resolve_llama_cli(s.get_str("llama_cli_path"), bin_dir),
        host=s.get_str("llama_host", "127.0.0.1"),
        port=s.get_int("llama_port", 8080),
        n_gpu_layers=s.get_int("n_gpu_layers", 0),
        threads=s.get_int("threads", -1),
        flash_attn=s.get_bool("flash_attn", False),
        cont_batching=s.get_bool("cont_batching", True),
    )


class ModelsPanel(ctk.CTkFrame):
    def __init__(self, master, active: ActiveConfig,
                 library: ProfileLibrary | None = None,
                 gguf_path_getter=None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)
        self.active = active
        self.library = library or ProfileLibrary()
        self._gguf_path_getter = gguf_path_getter or (lambda: "")

        self._sliders: dict[str, ctk.CTkSlider] = {}
        self._value_lbls: dict[str, ctk.CTkLabel] = {}
        self._suspend = False  # guard against feedback loops

        self.profile_var = ctk.StringVar(value="")
        self.status_msg = ctk.StringVar(value="")

        # Footer first so it never gets pushed off-screen.
        self._footer = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0)
        self._footer.pack(side="bottom", fill="x")
        self._content = ctk.CTkScrollableFrame(self, fg_color=theme.BG)
        self._content.pack(side="top", fill="both", expand=True)

        self._build_content()
        self._build_footer()

        self.active.subscribe(self._on_active_changed)
        self.after(80, self._sync_from_active)
        self.after(120, self._refresh_preview)

    # ── Content builders ───────────────────────────────────────────────────────

    def _heading(self, text: str):
        ctk.CTkLabel(self._content, text=text,
                     font=ctk.CTkFont(size=theme.fs_heading(), weight="bold"),
                     text_color=theme.ACCENT()).pack(anchor="w", padx=14, pady=(14, 4))

    def _build_content(self):
        self._heading("Sampling Parameters")
        grid = ctk.CTkFrame(self._content, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=(0, 4))
        grid.grid_columnconfigure((0, 1), weight=1, uniform="params")
        for i, key in enumerate(PARAM_KEYS):
            cell = ctk.CTkFrame(grid, fg_color=theme.SURFACE, corner_radius=8)
            cell.grid(row=i // 2, column=i % 2, sticky="ew", padx=6, pady=5)
            self._param_row(cell, key)

        self._heading("System Prompt")
        self.system_box = ctk.CTkTextbox(
            self._content, height=90, fg_color=theme.SURFACE2,
            font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
            text_color=theme.TEXT(), wrap="word",
        )
        self.system_box.pack(fill="x", padx=14, pady=(0, 6))
        self.system_box.bind("<KeyRelease>", self._on_prompt_edit)

        self._heading("Template (optional)")
        self.template_box = ctk.CTkTextbox(
            self._content, height=70, fg_color=theme.SURFACE2,
            font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
            text_color=theme.TEXT(), wrap="word",
        )
        self.template_box.pack(fill="x", padx=14, pady=(0, 6))
        self.template_box.bind("<KeyRelease>", self._on_prompt_edit)

        self._heading("Command Preview")
        self.preview_box = ctk.CTkTextbox(
            self._content, height=90, fg_color=theme.SURFACE,
            font=ctk.CTkFont(family="Consolas", size=theme.fs_small()),
            text_color=theme.MUTED(), wrap="word",
        )
        self.preview_box.pack(fill="x", padx=14, pady=(0, 10))
        self.preview_box.configure(state="disabled")

    def _param_row(self, parent, key: str):
        spec = PARAMETERS[key]
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(8, 0))
        ctk.CTkLabel(top, text=spec["label"],
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.TEXT()).pack(side="left")
        val_lbl = ctk.CTkLabel(top, text=str(spec["default"]),
                               font=ctk.CTkFont(size=theme.fs_small()),
                               text_color=theme.ACCENT())
        val_lbl.pack(side="right")
        self._value_lbls[key] = val_lbl

        desc = spec.get("desc", "")
        if desc:
            ctk.CTkLabel(top, text=desc, anchor="w",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(side="left", padx=(8, 0))

        steps = max(1, int(round((spec["max"] - spec["min"]) / spec["step"])))
        slider = ctk.CTkSlider(
            parent, from_=spec["min"], to=spec["max"], number_of_steps=steps,
            command=lambda v, k=key: self._on_slider(k, v),
        )
        slider.set(spec["default"])
        slider.pack(fill="x", padx=10, pady=(2, 10))
        self._sliders[key] = slider

    # ── Footer (library + export) ─────────────────────────────────────────────

    def _build_footer(self):
        # Library row
        lib = ctk.CTkFrame(self._footer, fg_color="transparent")
        lib.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(lib, text="Profile",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        self.profile_combo = ctk.CTkComboBox(
            lib, variable=self.profile_var, values=self._profile_values(),
            width=200, height=30, font=ctk.CTkFont(size=theme.fs_small()),
        )
        self.profile_combo.pack(side="left", padx=(8, 8))

        for text, cmd in (
            ("Load", self._load_profile), ("Save", self._save_profile),
            ("Save As", self._save_as_profile), ("Duplicate", self._duplicate_profile),
            ("Rename", self._rename_profile), ("Delete", self._delete_profile),
        ):
            ctk.CTkButton(lib, text=text, width=66, height=30,
                          fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                          command=cmd,
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(0, 4))

        # Export row
        exp = ctk.CTkFrame(self._footer, fg_color="transparent")
        exp.pack(fill="x", padx=12, pady=(2, 4))
        ctk.CTkButton(exp, text="Reset Defaults", width=110, height=28,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._reset_defaults,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(0, 8))
        ctk.CTkButton(exp, text="Export .ini…", width=100, height=28,
                      fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(),
                      command=self._export_ini,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(0, 6))
        ctk.CTkButton(exp, text="Export Modelfile…", width=130, height=28,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._export_modelfile,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(0, 6))
        ctk.CTkButton(exp, text="Copy Command", width=110, height=28,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._copy_command,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="left")

        ctk.CTkLabel(self._footer, textvariable=self.status_msg, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(fill="x", padx=14, pady=(0, 8))

    # ── Build current config ─────────────────────────────────────────────────

    def _current_cfg(self):
        return self.active.to_model_config(
            gguf_path=self._gguf_path_getter() or "", **_backend_kwargs()
        )

    # ── Slider / prompt handlers ──────────────────────────────────────────────

    def _on_slider(self, key: str, value: float):
        if self._suspend:
            return
        spec = PARAMETERS[key]
        val = int(round(value)) if spec["type"] is int else round(value, 3)
        self._value_lbls[key].configure(text=str(val))
        self.active.set_param(key, val, notify=False)
        self._refresh_preview()

    def _on_prompt_edit(self, _e=None):
        if self._suspend:
            return
        self.active.set_system_prompt(self.system_box.get("1.0", "end").rstrip("\n"), notify=False)
        self.active.set_template(self.template_box.get("1.0", "end").rstrip("\n"), notify=False)
        self._refresh_preview()

    # ── Sync from ActiveConfig (e.g. after profile load) ──────────────────────

    def _on_active_changed(self):
        try:
            self.after(0, self._sync_from_active)
        except Exception:
            pass

    def _sync_from_active(self):
        self._suspend = True
        try:
            for key in PARAM_KEYS:
                val = self.active.params.get(key, default_params()[key])
                if key in self._sliders:
                    self._sliders[key].set(val)
                    self._value_lbls[key].configure(text=str(val))
            self.system_box.delete("1.0", "end")
            self.system_box.insert("1.0", self.active.system_prompt or "")
            self.template_box.delete("1.0", "end")
            self.template_box.insert("1.0", self.active.template or "")
            if self.active.profile_name:
                self.profile_var.set(self.active.profile_name)
        finally:
            self._suspend = False
        self._refresh_preview()

    # ── Preview ──────────────────────────────────────────────────────────────

    def _refresh_preview(self):
        try:
            cfg = self._current_cfg()
            cmd = lc.build_command_str(cfg, mode=cfg.llama_mode)
        except Exception as e:
            cmd = f"(preview unavailable: {e})"
        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        self.preview_box.insert("1.0", cmd)
        self.preview_box.configure(state="disabled")

    # ── Library helpers ────────────────────────────────────────────────────────

    def _profile_values(self):
        items = self.library.list_profiles()
        return items if items else ["(no profiles)"]

    def _refresh_profile_list(self):
        self.profile_combo.configure(values=self._profile_values())

    def _selected_profile(self) -> str:
        name = self.profile_var.get().strip()
        return "" if (not name or name.startswith("(")) else name

    # ── Library actions ────────────────────────────────────────────────────────

    def _load_profile(self):
        name = self._selected_profile()
        if not name:
            self.status_msg.set("Select a profile to load.")
            return
        cfg = self.library.load(name)
        if cfg is None:
            messagebox.showerror("Load failed", f"Could not read profile '{name}'.", parent=self)
            return
        self.active.load_from(cfg)  # notifies → _sync_from_active
        self.status_msg.set(f"Loaded '{name}'.")

    def _save_profile(self):
        name = self._selected_profile() or self.active.profile_name
        if not name:
            self._save_as_profile()
            return
        self._do_save(name)

    def _save_as_profile(self):
        name = simpledialog.askstring("Save Profile As", "Profile name:", parent=self)
        if not name:
            return
        if self.library.exists(name) and not messagebox.askyesno(
                "Overwrite?", f"Profile '{name}' exists. Overwrite?", parent=self):
            return
        self._do_save(name)

    def _do_save(self, name: str):
        cfg = self._current_cfg()
        if self.library.save(name, cfg):
            self.active.profile_name = self.library.path_for(name).stem
            self._refresh_profile_list()
            self.profile_var.set(self.active.profile_name)
            self.status_msg.set(f"Saved '{name}'.")
        else:
            messagebox.showerror("Save failed", f"Could not save profile '{name}'.", parent=self)

    def _duplicate_profile(self):
        name = self._selected_profile()
        if not name:
            self.status_msg.set("Select a profile to duplicate.")
            return
        new = simpledialog.askstring("Duplicate Profile", "New name:", parent=self)
        if not new:
            return
        if self.library.duplicate(name, new):
            self._refresh_profile_list()
            self.profile_var.set(self.library.path_for(new).stem)
            self.status_msg.set(f"Duplicated to '{new}'.")
        else:
            messagebox.showerror("Duplicate failed",
                                 "Target may already exist.", parent=self)

    def _rename_profile(self):
        name = self._selected_profile()
        if not name:
            self.status_msg.set("Select a profile to rename.")
            return
        new = simpledialog.askstring("Rename Profile", "New name:", parent=self, initialvalue=name)
        if not new or new == name:
            return
        if self.library.rename(name, new):
            self._refresh_profile_list()
            self.profile_var.set(self.library.path_for(new).stem)
            self.active.profile_name = self.library.path_for(new).stem
            self.status_msg.set(f"Renamed to '{new}'.")
        else:
            messagebox.showerror("Rename failed", "Target may already exist.", parent=self)

    def _delete_profile(self):
        name = self._selected_profile()
        if not name:
            self.status_msg.set("Select a profile to delete.")
            return
        if not messagebox.askyesno("Delete?", f"Delete profile '{name}'?", parent=self):
            return
        if self.library.delete(name):
            self._refresh_profile_list()
            self.profile_var.set("")
            self.status_msg.set(f"Deleted '{name}'.")
        else:
            messagebox.showerror("Delete failed", f"Could not delete '{name}'.", parent=self)

    # ── Export actions ───────────────────────────────────────────────────────

    def _reset_defaults(self):
        self.active.reset_params()  # notifies → sync
        self.status_msg.set("Parameters reset to defaults.")

    def _export_ini(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".ini",
            filetypes=[("INI profile", "*.ini"), ("All files", "*.*")],
            initialfile=(self.active.profile_name or "profile") + ".ini",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(lc.to_ini(self._current_cfg()))
            self.status_msg.set(f"Exported INI → {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self)

    def _export_modelfile(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension="",
            filetypes=[("Modelfile", "*"), ("All files", "*.*")],
            initialfile="Modelfile",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(lc.to_modelfile(self._current_cfg()))
            self.status_msg.set(f"Exported Modelfile → {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self)

    def _copy_command(self):
        try:
            cmd = lc.build_command_str(self._current_cfg(), mode=self._current_cfg().llama_mode)
            self.clipboard_clear()
            self.clipboard_append(cmd)
            self.status_msg.set("Command copied to clipboard.")
        except Exception as e:
            self.status_msg.set(f"Copy failed: {e}")

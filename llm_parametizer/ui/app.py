"""
LLMParametizerApp — main application window.

Hosts a tabview with the Chat, Models, Tune, and RAG tabs; Settings is a modal
dialog opened from the Chat top bar.
"""

from __future__ import annotations

import time
from pathlib import Path

import customtkinter as ctk

from .. import __app_name__, __version__
from ..core.config import settings
from ..core import memory
from ..core.models import coerce_param, PARAMETERS
from ..core.state import ActiveConfig
from ..core.library import ProfileLibrary
from ..core.rag import RagStore
from . import theme
from .chat_panel import ChatPanel
from .models_panel import ModelsPanel
from .tune_panel import TunePanel
from .rag_panel import RagPanel
from .report_panel import ReportPanel
from .settings_dialog import SettingsDialog
from ..core import families
from ..core.metrics import Snapshot


def _safe_num(value):
    """Coerce to a finite number for safe IPC metrics, else None."""
    if value is None or value == "" or value == "—":
        return None
    try:
        n = float(str(value).replace("ms", "").strip())
    except (TypeError, ValueError):
        return None
    return n if n == n and n not in (float("inf"), float("-inf")) else None


class LLMParametizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(settings.get_str("appearance_mode", "dark"))
        ctk.set_default_color_theme(settings.get_str("color_theme", "dark-blue"))

        self.title(f"{__app_name__} v{__version__}")
        self.minsize(820, 560)
        self.configure(fg_color=theme.BG)
        self._restore_geometry()

        # Last run metrics reported by a peer (LLM Tester) over IPC.
        self._last_test_metrics = None

        self._build()
        self._register_ipc_handlers()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _restore_geometry(self) -> None:
        """Restore the saved window geometry if valid, else center a default."""
        saved = settings.get_str("window_geometry", "")
        if self._geometry_fits(saved):
            self.geometry(saved)
        else:
            self._center_window(1040, 700)

    def _geometry_fits(self, geo: str) -> bool:
        """True if a 'WxH+X+Y' string is well-formed and on-screen."""
        try:
            parts = geo.split("+")
            w, h = (int(v) for v in parts[0].split("x"))
            x, y = int(parts[1]), int(parts[2])
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            return 400 <= w <= sw and 300 <= h <= sh and -50 <= x <= sw - 100 and -10 <= y <= sh - 100
        except Exception:
            return False

    def _center_window(self, width: int, height: int) -> None:
        """Size the window (clamped to the screen) and center it."""
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(width, int(sw * 0.9))
        h = min(height, int(sh * 0.9))
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)   # slightly above center looks balanced
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build(self):
        # Shared state across tabs.
        self.active = ActiveConfig()
        self.library = ProfileLibrary()
        self.rag_store = RagStore()

        self._workflow_ready = False
        self._tab_buttons: dict[str, ctk.CTkButton] = {}

        # Custom separated tab bar (replaces the joined segmented button) — packed
        # first so it sits above the content; the hint bar is inserted below it
        # on demand via pack(before=self.tabs).
        self._tabbar = ctk.CTkFrame(self, fg_color=theme.BG)
        self._tabbar.pack(side="top", fill="x", padx=12, pady=(8, 2))
        self._hint_bar = ctk.CTkFrame(self, fg_color=theme.SURFACE2, corner_radius=8)

        self.tabs = ctk.CTkTabview(self, fg_color=theme.SURFACE)
        self.tabs.pack(side="top", fill="both", expand=True, padx=8, pady=(2, 8))

        # Tab order follows the user workflow: feel it out (Chat) → measure
        # (Tune) → fine-tune (Parameters) → validate with documents (RAG).
        chat_tab = self.tabs.add("Chat")
        self.chat_panel = ChatPanel(chat_tab, active=self.active, rag_store=self.rag_store)
        self.chat_panel.pack(fill="both", expand=True)
        # Settings now opens as a modal dialog from the Chat top bar.
        self.chat_panel.on_open_settings = self._open_settings
        self.chat_panel.on_model_changed = lambda: self._show_workflow_hint("Model changed")

        tune_tab = self.tabs.add("Tune")
        self.tune_panel = TunePanel(
            tune_tab, active=self.active, library=self.library,
            target_getter=self.chat_panel.current_chat_target,
            gguf_path_getter=self.chat_panel._selected_gguf_path,
            rag_store=self.rag_store,
        )
        self.tune_panel.pack(fill="both", expand=True)

        # "Parameters" is the manual parameter + system-prompt editor
        # (formerly labelled "Models").
        params_tab = self.tabs.add("Parameters")
        self.models_panel = ModelsPanel(
            params_tab, active=self.active, library=self.library,
            gguf_path_getter=self.chat_panel._selected_gguf_path,
        )
        self.models_panel.pack(fill="both", expand=True)

        rag_tab = self.tabs.add("RAG")
        self.rag_panel = RagPanel(rag_tab, store=self.rag_store)
        self.rag_panel.pack(fill="both", expand=True)
        self.rag_panel.on_switch_to_chat = lambda: self._select_tab("Chat")

        report_tab = self.tabs.add("Report")
        self.report_panel = ReportPanel(
            report_tab,
            capture_chat=self._capture_chat_snapshot,
            capture_tune=self._capture_tune_snapshot,
        )
        self.report_panel.pack(fill="both", expand=True)

        # Two categories: USE tools (Chat, RAG) and CONFIGURE tools (Tune,
        # Parameters), color-coded and divided in the tab bar.
        self._build_tab_bar([
            ("USE", theme.ACCENT, ["Chat", "RAG"]),
            ("CONFIGURE", theme.ACCENT2, ["Tune", "Parameters"]),
            ("DELIVER", lambda: theme.SUCCESS, ["Report"]),
        ])
        self._apply_tab_font()
        self._select_tab("Chat")

        # Hide the built-in joined segmented button AFTER tabs are added/selected
        # (each add()/set() re-grids it, which would undo an earlier hide).
        self._hide_native_tabbar()

        # Recommend re-testing when parameters change (sliders / apply from Tune).
        self.active.subscribe(lambda: self._show_workflow_hint("Parameters changed"))
        # Suppress hints during initial construction churn.
        self.after(1500, lambda: setattr(self, "_workflow_ready", True))

        self._settings_dialog = None

    # ── Custom tab bar + workflow hint ─────────────────────────────────────────

    def _build_tab_bar(self, groups: list[tuple]) -> None:
        """groups: list of (caption, color_fn, [tab_names]). Renders color-coded
        clusters separated by a divider, with a small category caption each."""
        self._tab_color: dict[str, str] = {}
        # Persistent "home" button pinned right — quick return to Chat (where the
        # server is started) from any utility/config tab.
        self._home_btn = ctk.CTkButton(
            self._tabbar, text="🏠 Chat", width=110, height=36, corner_radius=8,
            fg_color=theme.SURFACE2, hover_color=theme.BORDER, text_color=theme.ACCENT(),
            font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
            command=lambda: self._select_tab("Chat"))
        self._home_btn.pack(side="right", padx=(10, 0))
        for gi, (caption, color_fn, names) in enumerate(groups):
            color = color_fn()
            if gi > 0:
                # Vertical divider between categories (explicit height: CTkFrame
                # defaults to 200px, which would inflate the whole tab bar).
                ctk.CTkFrame(self._tabbar, width=2, height=28, fg_color=theme.BORDER).pack(
                    side="left", padx=(8, 12), pady=4)
            ctk.CTkLabel(self._tabbar, text=caption,
                         font=ctk.CTkFont(size=theme.fs_small(), weight="bold"),
                         text_color=color).pack(side="left", padx=(0, 8))
            for name in names:
                self._tab_color[name] = color
                btn = ctk.CTkButton(
                    self._tabbar, text=name, width=130, height=36, corner_radius=8,
                    fg_color=theme.SURFACE2, hover_color=theme.BORDER, text_color=color,
                    command=lambda n=name: self._select_tab(n),
                )
                btn.pack(side="left", padx=(0, 10))
                self._tab_buttons[name] = btn

    def _hide_native_tabbar(self) -> None:
        """Hide CTkTabview's built-in segmented button AND collapse the header
        rows it reserves (rows 0-2), which otherwise leave a tall empty gap."""
        try:
            self.tabs._segmented_button.grid_forget()
            for r in (0, 1, 2):
                self.tabs.grid_rowconfigure(r, minsize=0, weight=0)
        except Exception:
            pass

    def _select_tab(self, name: str) -> None:
        self.tabs.set(name)
        self._hide_native_tabbar()
        for n, btn in self._tab_buttons.items():
            color = self._tab_color.get(n, theme.ACCENT())
            if n == name:
                btn.configure(fg_color=color, text_color="#10141C")
            else:
                btn.configure(fg_color=theme.SURFACE2, text_color=color)
        # Auto-refresh backend status when switching to Tune
        if name == "Tune":
            try:
                self.tune_panel._update_target_status()
            except Exception:
                pass

    def _apply_tab_font(self) -> None:
        size = settings.get_int("tab_font_size", 15)
        font = ctk.CTkFont(size=size, weight="bold")
        for btn in self._tab_buttons.values():
            btn.configure(font=font)

    def _show_workflow_hint(self, reason: str = "Model or parameters changed") -> None:
        if not getattr(self, "_workflow_ready", False):
            return
        for w in self._hint_bar.winfo_children():
            w.destroy()
        msg = (f"{reason} — consider re-testing:  Chat  ·  Tune  ·  "
               f"Parameters  ·  RAG")
        ctk.CTkLabel(self._hint_bar, text=msg, anchor="w", justify="left",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.TEXT()).pack(side="left", padx=(12, 8), pady=6)
        ctk.CTkButton(self._hint_bar, text="✕", width=28, height=24,
                      fg_color="transparent", hover_color=theme.BORDER,
                      command=self._hint_bar.pack_forget,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=8)
        self._hint_bar.pack(side="top", fill="x", padx=12, pady=(0, 2), before=self.tabs)

    # ── Report snapshot capture ────────────────────────────────────────────────

    def _model_name(self) -> str:
        name = self.chat_panel.model_var.get()
        return "" if name in ("", "(no model)") else name

    def _current_goal(self) -> str:
        try:
            return self.tune_panel.goal_var.get()
        except Exception:
            return ""

    def _capture_chat_snapshot(self):
        """Snapshot of the current config + chat session performance."""
        sm = self.chat_panel.session_metrics
        if sm.count == 0:
            return None
        model = self._model_name()
        try:
            command = self.chat_panel.controller.command_preview()
        except Exception:
            command = ""
        return Snapshot(
            label="Baseline",
            model_name=model,
            family=families.detect_family(model),
            goal=self._current_goal(),
            params=dict(self.active.params),
            command=command,
            perf=sm.summary(),
        )

    def _capture_tune_snapshot(self):
        """Snapshot of the Tune tab's best/selected combo."""
        best = getattr(self.tune_panel, "_best", None)
        if best is None:
            return None
        m = self.tune_panel._combo_metrics(best)
        params = dict(self.active.params)
        params.update(best.params)
        model = self._model_name()
        try:
            command = self.chat_panel.controller.command_preview()
        except Exception:
            command = ""
        perf = {}
        if m.get("tps") is not None:
            perf["avg_tps"] = round(m["tps"], 1)
        if m.get("ttft") is not None:
            perf["avg_ttft"] = round(m["ttft"], 2)
        quality = {
            "score": best.avg_score,
            "repetition": round(m.get("rep", 0.0), 3),
            "refusals": m.get("refusals", 0),
        }
        return Snapshot(
            label="Optimized",
            model_name=model,
            family=families.detect_family(model),
            goal=self._current_goal(),
            params=params,
            command=command,
            perf=perf,
            quality=quality,
        )

    def _open_settings(self):
        # Reuse an existing open dialog if present.
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.focus()
            return
        self._settings_dialog = SettingsDialog(self, on_saved=self._on_settings_saved)

    def _on_settings_saved(self):
        # Refresh chat engine/model list to reflect new paths.
        try:
            self.chat_panel._on_engine_change()
        except Exception:
            pass
        # Refresh context meter to reflect new n_ctx value.
        try:
            self.chat_panel.refresh_context_limit()
        except Exception:
            pass
        # Apply any tab font-size change immediately.
        try:
            self._apply_tab_font()
        except Exception:
            pass

    # ── IPC handlers (Shield / LLM Tester integration) ─────────────────────────

    def _profile_display_name(self) -> str:
        """Clean profile name for IPC — never a local path (rule 3)."""
        raw = self.active.profile_name or ""
        if not raw:
            return ""
        # profile_name may be stored as a path; expose only the stem.
        return Path(raw).stem if ("/" in raw or "\\" in raw) else raw

    def _safe_status(self) -> dict:
        """Allow-listed safe status fields shared by get_status and /ipc/status.

        Exposes only metrics/metadata — never prompts, templates, RAG text or
        local paths (IPC security rule 3). VRAM is included only when this app
        actually detects it (rule 6).
        """
        ctrl = self.chat_panel.controller
        model = self.chat_panel.model_var.get()
        status = {
            "running": ctrl.is_running,
            "reachable": ctrl.chat_reachable(),
            "engine": self.chat_panel.engine_var.get(),
            "model": "" if model in ("", "(no model)") else model,
            "profileName": self._profile_display_name(),
            "params": dict(self.active.params),
            "updatedAt": time.time(),
        }
        # VRAM — Parametizer is the sole source (rule 6). Omit entirely if absent.
        vram = memory.vram_info()
        if vram.available and vram.total_bytes:
            status["vram"] = {
                "name": vram.name,
                "total_gb": round(vram.total_bytes / 1024 ** 3, 1),
                "used_gb": round(vram.used_bytes / 1024 ** 3, 1),
                "free_gb": round(vram.free_bytes / 1024 ** 3, 1),
            }
            cap_mb = settings.get_int("vram_cap_mb", 0)
            if cap_mb > 0:
                status["vramCap"] = round(cap_mb / 1024, 1)
        # Last run metrics reported by a peer (e.g. LLM Tester), if any.
        last = getattr(self, "_last_test_metrics", None)
        if last:
            status["lastTestMetrics"] = last
        return status

    # Profile fields that must never leave the app over IPC (rule 3).
    _UNSAFE_PROFILE_KEYS = {
        "system_prompt", "template", "extra_llama_args", "swap_models",
    }

    @classmethod
    def _safe_profile(cls, profile: dict) -> dict:
        """Strip raw prompts/templates, local paths and secrets from a profile."""
        safe = {}
        for key, value in (profile or {}).items():
            kl = key.lower()
            if key in cls._UNSAFE_PROFILE_KEYS:
                continue
            if "path" in kl or "secret" in kl or "token" in kl or "key" in kl:
                continue
            safe[key] = value
        return safe

    def _register_ipc_handlers(self):
        from ..core import ipc

        def _get_config(_data: dict) -> dict:
            return dict(self.active.params)

        def _set_config(data: dict) -> dict:
            params = data.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("'params' must be an object.")
            applied, rejected = {}, []
            for key, value in params.items():
                # Only known parameter names are accepted (rule 8).
                if key not in PARAMETERS:
                    rejected.append(key)
                    continue
                # Coerce to the declared type, then clamp to the allowed range.
                coerced = coerce_param(key, value)
                spec = PARAMETERS[key]
                try:
                    coerced = max(spec["min"], min(spec["max"], coerced))
                except TypeError:
                    pass
                self.active.params[key] = coerced
                applied[key] = coerced
            if applied:
                self.active.notify()
            return {"applied": applied, "rejected": rejected}

        def _get_status(_data: dict) -> dict:
            return self._safe_status()

        def _get_model_info(_data: dict) -> dict:
            # Safe metadata only — no system prompt / template / paths (rule 3).
            return {
                "engine": self.chat_panel.engine_var.get(),
                "model": self._safe_status()["model"],
                "profileName": self._profile_display_name(),
                "n_ctx": settings.get_int("n_ctx", 4096),
                "n_gpu_layers": settings.get_int("n_gpu_layers", 0),
                "params": dict(self.active.params),
            }

        def _export_profile(data: dict) -> dict:
            name = data.get("name", "")
            if not name:
                names = self.library.list_profiles()
                return {"profiles": names}
            profile = self.library.load_profile(name)
            if profile is None:
                raise ValueError(f"Profile '{name}' not found.")
            return {"name": name, "profile": self._safe_profile(profile)}

        def _report_metrics(data: dict) -> dict:
            # Inbound safe run metrics from LLM Tester. Numbers + short labels
            # only — never prompt/response text (rule 3).
            m = data.get("metrics", data) or {}
            safe = {
                "source": str(m.get("source", "peer"))[:60],
                "model": str(m.get("modelName", m.get("model", "")))[:120],
                "tps": _safe_num(m.get("tps")),
                "ttftMs": _safe_num(m.get("ttftMs", m.get("ttft"))),
                "totalTokens": _safe_num(m.get("totalTokens")),
                "status": "failure" if m.get("status") == "failure" else "success",
                "receivedAt": time.time(),
            }
            self._last_test_metrics = safe
            return {"stored": True}

        def _get_metrics(_data: dict) -> dict:
            return {"lastTestMetrics": getattr(self, "_last_test_metrics", None)}

        ipc.register_handler("get_config", _get_config)
        ipc.register_handler("set_config", _set_config)
        ipc.register_handler("get_status", _get_status)
        ipc.register_handler("get_model_info", _get_model_info)
        ipc.register_handler("export_profile", _export_profile)
        ipc.register_handler("report_metrics", _report_metrics)
        ipc.register_handler("get_metrics", _get_metrics)
        # Expose safe status on the GET /ipc/status probe for peer detection.
        ipc.register_status_provider(self._safe_status)

    def _on_close(self):
        try:
            if self.state() == "normal":   # don't persist a maximized/zoomed size
                settings.set("window_geometry", self.geometry())
                settings.save()
        except Exception:
            pass
        # Shut down IPC bridge.
        try:
            from ..core import ipc
            ipc.stop()
        except Exception:
            pass
        try:
            self.chat_panel.shutdown()
        except Exception:
            pass
        # Safety net: kill any orphaned llama processes that survived shutdown.
        try:
            self.chat_panel.controller.kill_orphans()
        except Exception:
            pass
        self.destroy()


def run():
    app = LLMParametizerApp()
    app.mainloop()

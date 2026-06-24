"""
TunePanel — AutoTune UI.

Pick which sampling parameters to sweep (with candidate values), a prompt suite,
a goal and optional required keyword, then run combos against the already-running
backend. Results are scored and ranked live; the best combo can be applied to the
shared ActiveConfig or saved as a profile.

The backend must already be running (started from the Chat tab). num_ctx is held
fixed during a sweep (changing it needs a server restart).
"""

from __future__ import annotations

import customtkinter as ctk
from tkinter import messagebox

from . import theme
from .widgets import StatusPill, MetricsBar
from ..core.models import PARAMETERS, coerce_param
from ..core.state import ActiveConfig
from ..core.library import ProfileLibrary
from ..core import tuning, families
from ..core.tuning import SweepSpec, ComboScore
from ..tune.runner import TuneRunner


_GOALS = ["balanced", "accuracy", "creative", "concise"]

# Plain-language explanation of each goal (shown live under the picker).
_GOAL_DESC = {
    "balanced": "Rewards clear, on-topic, non-repetitive answers, plus a small speed bonus. No length preference.",
    "accuracy": "Aimed at factual replies. Tip: set a Required keyword so a run only scores well if the answer contains it.",
    "creative": "Lightly favors longer, more varied answers.",
    "concise":  "Lightly favors shorter answers and penalizes long, rambling output.",
}

# What the 0-100 score measures (shown under Results).
_SCORE_HELP = ("Score 0-100: starts at 90 for a clean answer, then "
               "− repetition, − refusals, − too-short, ± keyword, + speed.")

_BLACK = "#10141C"   # readable dark text for bright (green/blue) buttons


class TunePanel(ctk.CTkFrame):
    def __init__(self, master, active: ActiveConfig,
                 library: ProfileLibrary | None = None,
                 target_getter=None, gguf_path_getter=None, rag_store=None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)
        self.active = active
        self.library = library or ProfileLibrary()
        self.rag_store = rag_store
        self._target_getter = target_getter or (lambda: (None, "127.0.0.1", 8080, "", False, "No target."))
        self._gguf_path_getter = gguf_path_getter or (lambda: "")

        self.runner = TuneRunner(
            on_progress=lambda d, t, label: self.after(0, self._on_progress, d, t, label),
            on_combo=lambda c: self.after(0, self._on_combo, c),
            on_finished=lambda results: self.after(0, self._on_finished, results),
            on_error=lambda m: self.after(0, self._on_error, m),
        )

        self._sweep_vars: dict[str, ctk.BooleanVar] = {}
        self._sweep_entries: dict[str, ctk.CTkEntry] = {}
        self._result_rows: list[ComboScore] = []
        self._best: ComboScore | None = None

        self.goal_var = ctk.StringVar(value="balanced")
        self.goal_desc = ctk.StringVar(value=_GOAL_DESC["balanced"])
        self.keyword_var = ctk.StringVar(value="")
        self.use_rag_var = ctk.BooleanVar(value=False)
        self.target_info = ctk.StringVar(value="Checking backend…")
        self.status_msg = ctk.StringVar(value="Start a model in the Chat tab, then run a test here.")

        self._footer = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0)
        self._footer.pack(side="bottom", fill="x")
        self._content = ctk.CTkScrollableFrame(self, fg_color=theme.BG)
        self._content.pack(side="top", fill="both", expand=True)

        self._build_content()
        self._build_footer()
        self._update_target_status()

    # ── Content ──────────────────────────────────────────────────────────────

    def _heading(self, text: str, sub: str = ""):
        ctk.CTkLabel(self._content, text=text,
                     font=ctk.CTkFont(size=theme.fs_heading(), weight="bold"),
                     text_color=theme.ACCENT()).pack(anchor="w", padx=14, pady=(14, 0))
        if sub:
            ctk.CTkLabel(self._content, text=sub, anchor="w",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(anchor="w", padx=14, pady=(0, 4))

    def _build_content(self):
        # ── Backend / model status banner ──────────────────────────────────────
        banner = ctk.CTkFrame(self._content, fg_color=theme.SURFACE, corner_radius=8)
        banner.pack(fill="x", padx=14, pady=(12, 4))
        self.target_pill = StatusPill(banner, label="Unknown")
        self.target_pill.pack(side="left", padx=(10, 10), pady=8)
        ctk.CTkLabel(banner, textvariable=self.target_info, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.TEXT()).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(banner, text="↻ Refresh", width=90, height=28,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._update_target_status,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=10)

        self._heading(
            "Parameters to Test",
            "A 'test' tries every combination of the values you check below and scores each. "
            "Check a parameter, then edit its comma-separated candidate values.")
        grid = ctk.CTkFrame(self._content, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=(0, 4))
        grid.grid_columnconfigure(2, weight=1)
        for i, key in enumerate(tuning.SWEEPABLE):
            var = ctk.BooleanVar(value=(key == "temperature"))
            self._sweep_vars[key] = var
            cb = ctk.CTkCheckBox(grid, text="", variable=var, width=24,
                                 command=self._refresh_combo_count)
            cb.grid(row=i, column=0, padx=(8, 2), pady=5)
            ctk.CTkLabel(grid, text=PARAMETERS[key]["label"], width=120, anchor="w",
                         font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                         text_color=theme.TEXT()).grid(row=i, column=1, sticky="w", padx=(0, 6))
            ent = ctk.CTkEntry(grid, fg_color=theme.SURFACE2, height=30,
                               font=ctk.CTkFont(family="Consolas", size=theme.fs_small()))
            ent.insert(0, ", ".join(str(v) for v in tuning.DEFAULT_CANDIDATES[key]))
            ent.grid(row=i, column=2, sticky="ew", padx=(0, 10), pady=5)
            ent.bind("<KeyRelease>", lambda _e: self._refresh_combo_count())
            self._sweep_entries[key] = ent

        self.combo_lbl = ctk.CTkLabel(
            self._content, text="", anchor="w",
            font=ctk.CTkFont(size=theme.fs_small()), text_color=theme.MUTED())
        self.combo_lbl.pack(anchor="w", padx=14, pady=(0, 4))

        self._heading("Test Prompts", "One prompt per line. Every value combination is run against all of these.")
        self.prompts_box = ctk.CTkTextbox(
            self._content, height=90, fg_color=theme.SURFACE2,
            font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
            text_color=theme.TEXT(), wrap="word")
        self.prompts_box.pack(fill="x", padx=14, pady=(0, 6))
        self.prompts_box.insert("1.0", "\n".join(tuning.DEFAULT_PROMPTS))

        self._heading("How Answers Are Scored")
        opts = ctk.CTkFrame(self._content, fg_color="transparent")
        opts.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkLabel(opts, text="Goal", font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        ctk.CTkOptionMenu(opts, variable=self.goal_var, values=_GOALS, width=130, height=30,
                          command=self._on_goal_change,
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(6, 16))
        ctk.CTkLabel(opts, text="Required keyword (optional)",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        ctk.CTkEntry(opts, textvariable=self.keyword_var, width=180, height=30,
                     fg_color=theme.SURFACE2,
                     font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(6, 0))
        ctk.CTkLabel(self._content, textvariable=self.goal_desc, anchor="w", justify="left",
                     wraplength=760, font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.TEXT()).pack(anchor="w", padx=14, pady=(0, 8))

        rag_row = ctk.CTkFrame(self._content, fg_color="transparent")
        rag_row.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkSwitch(rag_row, text="Use RAG context (documents from the RAG tab)",
                      variable=self.use_rag_var,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      text_color=theme.TEXT()).pack(side="left")
        ctk.CTkLabel(rag_row, text="  Injects retrieved excerpts per prompt so configs are scored on "
                                  "document-grounded answers.",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(side="left")

        self._heading("Results", _SCORE_HELP)
        self.results_frame = ctk.CTkScrollableFrame(self._content, fg_color=theme.SURFACE,
                                                    height=170)
        self.results_frame.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkLabel(self._content, text="Selected answer (what the score is based on)",
                     anchor="w", font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(anchor="w", padx=14, pady=(2, 0))
        self.details = ctk.CTkTextbox(self._content, height=150, fg_color=theme.SURFACE2,
                                      font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
                                      text_color=theme.TEXT(), wrap="word")
        self.details.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.details.insert("1.0", "Run a test, then select a result to see the model's actual "
                                   "answer and the metric breakdown behind its score.")
        self.details.configure(state="disabled")

        self._refresh_combo_count()

    # ── Footer ───────────────────────────────────────────────────────────────

    def _set_enabled(self, btn, enabled: bool) -> None:
        """Enable with the button's colour palette; disable with a muted surface
        background and muted text so the label stays readable (not gray-on-color)."""
        color, tcolor = self._btn_palette.get(btn, (theme.ACCENT(), "#FFFFFF"))
        if enabled:
            btn.configure(state="normal", fg_color=color, text_color=tcolor)
        else:
            btn.configure(state="disabled", fg_color=theme.SURFACE2, text_color=theme.MUTED())

    def _build_footer(self):
        row = ctk.CTkFrame(self._footer, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(8, 2))

        _bfont = ctk.CTkFont(size=theme.fs_button(), weight="bold")
        self.run_btn = ctk.CTkButton(row, text="▶ Run Test", width=120, height=32,
                                     hover_color="#1F8B4C", command=self._on_run, font=_bfont)
        self.run_btn.pack(side="left", padx=(0, 8))
        self.cancel_btn = ctk.CTkButton(row, text="■ Cancel", width=90, height=32,
                                        hover_color="#A03228", command=self._on_cancel, font=_bfont)
        self.cancel_btn.pack(side="left", padx=(0, 16))
        self.apply_btn = ctk.CTkButton(row, text="Apply Best", width=100, height=32,
                                       hover_color=theme.ACCENT2(), command=self._apply_best, font=_bfont)
        self.apply_btn.pack(side="left", padx=(0, 6))
        self.save_btn = ctk.CTkButton(row, text="Save Best as Profile…", width=170, height=32,
                                      hover_color=theme.ACCENT2(), command=self._save_best, font=_bfont)
        self.save_btn.pack(side="left")

        # Enabled palette per button: (background, text). Disabled buttons fall
        # back to a muted surface bg + muted text (readable + clearly inactive).
        self._btn_palette = {
            self.run_btn: (theme.SUCCESS, _BLACK),
            self.cancel_btn: (theme.DANGER, "#FFFFFF"),
            self.apply_btn: (theme.ACCENT(), _BLACK),
            self.save_btn: (theme.ACCENT(), _BLACK),
        }
        self._set_enabled(self.run_btn, True)
        self._set_enabled(self.cancel_btn, False)
        self._set_enabled(self.apply_btn, False)
        self._set_enabled(self.save_btn, False)

        # Metrics bar for the currently selected/best combo. Hidden until a
        # combo is selected — no blank metrics before a test has run.
        self.metrics_bar = MetricsBar(self._footer, title="Best:")

        prow = ctk.CTkFrame(self._footer, fg_color="transparent")
        prow.pack(fill="x", padx=14, pady=(4, 2))
        self._prow = prow
        self.spinner_lbl = ctk.CTkLabel(prow, text="", width=18,
                                        font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                                        text_color=theme.ACCENT())
        self.spinner_lbl.pack(side="left", padx=(0, 4))
        self.progress = ctk.CTkProgressBar(
            prow, height=14, corner_radius=7,
            progress_color=theme.ACCENT(), fg_color=theme.SURFACE2)
        self.progress.set(0)
        self.progress.pack(side="left", fill="x", expand=True)
        self.percent_lbl = ctk.CTkLabel(prow, text="0%", width=44,
                                        font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                                        text_color=theme.TEXT())
        self.percent_lbl.pack(side="left", padx=(8, 0))

        # Spinner animation state
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._spinner_running = False

        ctk.CTkLabel(self._footer, textvariable=self.status_msg, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(fill="x", padx=14, pady=(0, 8))

    # ── Spec assembly ─────────────────────────────────────────────────────────

    def _parse_candidates(self, key: str) -> list:
        spec = PARAMETERS[key]
        raw = self._sweep_entries[key].get()
        out = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(coerce_param(key, tok))
            except Exception:
                continue
        return out

    def _build_spec(self) -> SweepSpec:
        param_values = {}
        for key, var in self._sweep_vars.items():
            if var.get():
                vals = self._parse_candidates(key)
                if vals:
                    param_values[key] = vals
        prompts = [ln.strip() for ln in self.prompts_box.get("1.0", "end").splitlines() if ln.strip()]
        return SweepSpec(
            param_values=param_values,
            base_params=dict(self.active.params),
            prompts=prompts,
            system_prompt=self.active.system_prompt,
            goal=self.goal_var.get(),
            keyword=self.keyword_var.get().strip(),
            model_hint=self._gguf_path_getter() or "",
            use_rag=bool(self.use_rag_var.get()),
        )

    def _refresh_combo_count(self):
        spec = self._build_spec()
        n = spec.combo_count()
        prompts = len(spec.prompts) or 0
        fam = families.detect_family(spec.model_hint)
        capped = min(n, tuning.MAX_COMBOS)
        msg = f"{capped} value combination(s) × {prompts} prompt(s) = {capped * prompts} model call(s)."
        if n > tuning.MAX_COMBOS:
            msg += f"  (limited to {tuning.MAX_COMBOS} combinations)"
        if fam != "unknown":
            msg += f"  · detected family: {fam}"
        self.combo_lbl.configure(text=msg)

    def _on_goal_change(self, *_):
        self.goal_desc.set(_GOAL_DESC.get(self.goal_var.get(), ""))

    def _update_target_status(self):
        backend, host, port, model, ok, reason = self._target_getter()
        bname = "Ollama" if getattr(backend, "value", "") == "ollama" else "llama.cpp"
        if ok:
            self.target_pill.set_state("running")
            who = model or "loaded model"
            self.target_info.set(f"Ready · {bname} · {who} · {host}:{port}")
        else:
            self.target_pill.set("Not Ready", fg=theme.WARNING, text_color="#1A1A1A")
            self.target_info.set(reason or "No backend running. Start one in the Chat tab.")

    # ── Progress spinner ──────────────────────────────────────────────────────

    def _start_spinner(self):
        self._spinner_running = True
        self._tick_spinner()

    def _stop_spinner(self):
        self._spinner_running = False
        self.spinner_lbl.configure(text="")

    def _tick_spinner(self):
        if not self._spinner_running:
            return
        self.spinner_lbl.configure(text=self._spinner_frames[self._spinner_idx])
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self.after(100, self._tick_spinner)

    def _set_percent(self, frac: float):
        pct = max(0, min(100, int(round(frac * 100))))
        self.progress.set(frac)
        self.percent_lbl.configure(text=f"{pct}%")

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _on_run(self):
        if self.runner.is_running:
            return
        self._update_target_status()
        backend, host, port, model, ok, reason = self._target_getter()
        if not ok:
            messagebox.showwarning("Backend not ready", reason, parent=self)
            return

        spec = self._build_spec()
        if not spec.param_values:
            messagebox.showwarning("Nothing to test",
                                   "Check at least one parameter to test.", parent=self)
            return
        if not spec.prompts:
            messagebox.showwarning("No prompts", "Add at least one test prompt.", parent=self)
            return

        n = min(spec.combo_count(), tuning.MAX_COMBOS) * len(spec.prompts)
        if not messagebox.askyesno("Run test?",
                                   f"This will send {n} request(s) to the running model "
                                   f"and may take a while. Continue?",
                                   parent=self):
            return

        self._clear_results()
        self._best = None
        self._set_enabled(self.apply_btn, False)
        self._set_enabled(self.save_btn, False)
        self.run_btn.configure(text="● Running…")
        self._set_enabled(self.run_btn, False)
        self._set_enabled(self.cancel_btn, True)
        self._set_percent(0)
        self._start_spinner()
        self.status_msg.set("Running test…")

        self.runner.start(spec, backend, host=host, port=port, model=model,
                          rag_store=self.rag_store)

    def _on_cancel(self):
        if self.runner.is_running:
            self.runner.cancel()
            # Immediately reset UI — don't wait for the thread to finish
            self._stop_spinner()
            self._set_percent(0)
            self.run_btn.configure(text="▶ Run Test")
            self._set_enabled(self.run_btn, True)
            self._set_enabled(self.cancel_btn, False)
            self.status_msg.set("Cancelled.")

    # ── Runner callbacks (UI thread) ────────────────────────────────────────────

    def _on_progress(self, done: int, total: int, label: str):
        if self.runner._cancel.is_set():
            return  # ignore late callbacks after cancel
        self._set_percent(done / total if total else 0)
        self.status_msg.set(f"{label}  ({done}/{total})")

    def _on_combo(self, combo: ComboScore):
        self._result_rows.append(combo)
        self._render_results(tuning.rank(self._result_rows))

    def _on_finished(self, results: list):
        was_cancelled = self.runner._cancel.is_set()
        self._stop_spinner()
        self._result_rows = list(results)
        self._render_results(results)
        self.run_btn.configure(text="▶ Run Test")
        self._set_enabled(self.run_btn, True)
        self._set_enabled(self.cancel_btn, False)
        if was_cancelled:
            self._set_percent(0)
            if results:
                self._best = results[0]
                self._select_combo(self._best)
                self.status_msg.set(f"Cancelled. Partial best: {self._best.swept_summary}  (score {self._best.avg_score})")
            else:
                self.status_msg.set("Cancelled.")
        else:
            self._set_percent(1.0)
            if results:
                self._best = results[0]
                self._select_combo(self._best)
                self.status_msg.set(f"Done. Best: {self._best.swept_summary}  (score {self._best.avg_score})")
            else:
                self.status_msg.set("No results.")

    def _on_error(self, msg: str):
        self._stop_spinner()
        self.run_btn.configure(text="▶ Run Test")
        self._set_enabled(self.run_btn, True)
        self._set_enabled(self.cancel_btn, False)
        self.status_msg.set(f"Error: {msg}")
        messagebox.showerror("Test failed", msg, parent=self)

    # ── Results table ───────────────────────────────────────────────────────────

    def _clear_results(self):
        self._result_rows = []
        for w in self.results_frame.winfo_children():
            w.destroy()

    def _render_results(self, ranked: list):
        for w in self.results_frame.winfo_children():
            w.destroy()
        for i, combo in enumerate(ranked):
            row = ctk.CTkFrame(self.results_frame, fg_color=theme.SURFACE2, corner_radius=6)
            row.pack(fill="x", padx=4, pady=2)
            badge = theme.SUCCESS if i == 0 else theme.MUTED()
            ctk.CTkLabel(row, text=f"#{i + 1}", width=30,
                         font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                         text_color=badge).pack(side="left", padx=(8, 2), pady=4)
            ctk.CTkLabel(row, text=f"{combo.avg_score:.0f}", width=40,
                         font=ctk.CTkFont(size=theme.fs_base() + 2, weight="bold"),
                         text_color=theme.ACCENT()).pack(side="left", padx=(0, 8))

            mid = ctk.CTkFrame(row, fg_color="transparent")
            mid.pack(side="left", fill="x", expand=True)
            summary = combo.swept_summary or "(base params)"
            ctk.CTkLabel(mid, text=summary, anchor="w",
                         font=ctk.CTkFont(family="Consolas", size=theme.fs_small()),
                         text_color=theme.TEXT()).pack(anchor="w")
            ctk.CTkLabel(mid, text=self._metrics_line(combo), anchor="w",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(anchor="w")

            ctk.CTkButton(row, text="View", width=58, height=26,
                          fg_color=theme.SURFACE, hover_color=theme.BORDER,
                          command=lambda c=combo: self._select_combo(c),
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=6, pady=4)

    def _combo_metrics(self, combo: ComboScore) -> dict:
        runs = [r for r in combo.runs if r.error is None]
        if not runs:
            return {"words": 0, "rep": 0.0, "refusals": combo.errors, "tps": None, "ttft": None}
        words = sum(r.metrics.get("words", 0) for r in runs) / len(runs)
        rep = sum(r.metrics.get("repetition", 0.0) for r in runs) / len(runs)
        refusals = sum(1 for r in runs if r.metrics.get("refusal"))
        tpss = [r.metrics.get("tokens_per_second") for r in runs
                if r.metrics.get("tokens_per_second")]
        tps = sum(tpss) / len(tpss) if tpss else None
        ttfts = [r.metrics.get("ttft") for r in runs if r.metrics.get("ttft") is not None]
        ttft = sum(ttfts) / len(ttfts) if ttfts else None
        return {"words": words, "rep": rep, "refusals": refusals, "tps": tps, "ttft": ttft}

    def _metrics_line(self, combo: ComboScore) -> str:
        m = self._combo_metrics(combo)
        parts = [f"~{m['words']:.0f} words", f"repetition {m['rep']:.2f}",
                 f"{m['refusals']} refusal(s)"]
        if m["tps"]:
            parts.append(f"{m['tps']:.1f} tok/s")
        if m["ttft"] is not None:
            parts.append(f"TTFT {m['ttft']:.2f}s")
        if combo.errors:
            parts.append(f"{combo.errors} error(s)")
        return "  ·  ".join(parts)

    def _select_combo(self, combo: ComboScore):
        self._best = combo
        self._set_enabled(self.apply_btn, True)
        self._set_enabled(self.save_btn, True)
        self.status_msg.set(f"Selected: {combo.swept_summary}  (score {combo.avg_score})")
        m = self._combo_metrics(combo)
        self.metrics_bar.set_metrics(
            ttft=round(m["ttft"], 2) if m["ttft"] is not None else None,
            tps=round(m["tps"], 1) if m["tps"] else None,
            tokens=None,
            elapsed=None)
        if self.metrics_bar.has_data and not self.metrics_bar.winfo_ismapped():
            self.metrics_bar.pack(fill="x", padx=14, pady=(4, 2), before=self._prow)
        elif not self.metrics_bar.has_data:
            self.metrics_bar.pack_forget()
        self._show_details(combo)

    def _show_details(self, combo: ComboScore):
        lines = []
        lines.append(f"Parameters: {combo.swept_summary or '(base params)'}")
        lines.append(f"Average score: {combo.avg_score:.1f} / 100     [ {self._metrics_line(combo)} ]")
        lines.append("")
        for r in combo.runs:
            if r.error:
                lines.append(f"PROMPT: {r.prompt}")
                lines.append(f"  ERROR: {r.error}")
            else:
                out = (r.output or "").strip()
                if len(out) > 600:
                    out = out[:600] + " …"
                lines.append(f"PROMPT: {r.prompt}   (score {r.score:.0f})")
                lines.append(f"  {out}")
            lines.append("")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")

    # ── Apply / save ─────────────────────────────────────────────────────────

    def _apply_best(self):
        if not self._best:
            return
        # Only the tested keys should overwrite the active config.
        swept = {k: self._best.params[k] for k in tuning.SWEEPABLE if k in self._best.params}
        self.active.set_params(swept)   # notifies → Models/Chat update
        self.status_msg.set(f"Applied to active config: {self._best.swept_summary}")

    def _save_best(self):
        if not self._best:
            return
        name = self._ask_profile_name()
        if not name:
            return
        self._apply_best()
        cfg = self.active.to_model_config(gguf_path=self._gguf_path_getter() or "")
        if self.library.save(name, cfg):
            path = self.library.path_for(name)
            self.status_msg.set(f"Saved profile to: {path}")
            messagebox.showinfo("Profile saved", f"Saved '{name}' to:\n\n{path}", parent=self)
        else:
            messagebox.showerror("Save failed", f"Could not save '{name}'.", parent=self)

    def _ask_profile_name(self) -> str | None:
        """Themed, readable name prompt that shows where the profile will be saved."""
        dlg = ctk.CTkToplevel(self)
        dlg.title("Save Profile")
        dlg.geometry("520x230")
        dlg.transient(self.winfo_toplevel())
        dlg.configure(fg_color=theme.SURFACE)
        result = {"name": None}

        ctk.CTkLabel(dlg, text="Save tuned settings as a profile",
                     font=ctk.CTkFont(size=theme.fs_title(), weight="bold"),
                     text_color=theme.TEXT()).pack(anchor="w", padx=20, pady=(18, 4))
        ctk.CTkLabel(dlg, text="Profile name", anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(anchor="w", padx=20)
        entry = ctk.CTkEntry(dlg, height=36, fg_color=theme.SURFACE2,
                             font=ctk.CTkFont(size=theme.fs_base()))
        entry.pack(fill="x", padx=20, pady=(2, 10))
        entry.focus_set()

        ctk.CTkLabel(dlg, text=f"Will be saved to:\n{self.library.dir}", anchor="w",
                     justify="left", font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(anchor="w", padx=20, pady=(0, 10))

        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(0, 16))

        def _ok():
            result["name"] = entry.get().strip() or None
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        ctk.CTkButton(btns, text="Save", width=100, height=34,
                      fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(), text_color=_BLACK,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      command=_ok).pack(side="right")
        ctk.CTkButton(btns, text="Cancel", width=90, height=34,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=_cancel).pack(side="right", padx=(0, 8))
        entry.bind("<Return>", lambda _e: _ok())
        dlg.bind("<Escape>", lambda _e: _cancel())

        dlg.grab_set()
        self.wait_window(dlg)
        return result["name"]

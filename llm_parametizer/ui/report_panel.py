"""
ReportPanel — capture before/after snapshots and export a client deliverable.

The consultant captures one or more configuration snapshots (typically a
"Baseline" and an "Optimized"), each carrying the params, launch command and
measured performance/quality metrics. The panel renders a before/after report
(Markdown or plain text) and exports it to a file the consultant hands to a
client.

Snapshots are supplied by the app via two getters:
  - capture_chat() -> Snapshot | None   (current model + chat session metrics)
  - capture_tune() -> Snapshot | None   (best combo from the Tune tab)
"""

from __future__ import annotations

import customtkinter as ctk
from tkinter import filedialog, messagebox

from . import theme
from ..core import report
from ..core.metrics import Snapshot, hardware_summary


_LABELS = ["Baseline", "Optimized", "Candidate A", "Candidate B"]
_BLACK = "#10141C"


class ReportPanel(ctk.CTkFrame):
    def __init__(self, master, capture_chat=None, capture_tune=None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)
        self.capture_chat = capture_chat or (lambda: None)
        self.capture_tune = capture_tune or (lambda: None)
        self.snapshots: list[Snapshot] = []

        self.client_var = ctk.StringVar(value="")
        self.label_var = ctk.StringVar(value=_LABELS[0])
        self.fmt_var = ctk.StringVar(value="Markdown")
        self.notes_var = ctk.StringVar(value="")
        self.status_msg = ctk.StringVar(value="Capture a configuration to begin.")

        self._build()

    # ── Layout ──────────────────────────────────────────────────────────────────

    def _heading(self, text: str, sub: str = ""):
        ctk.CTkLabel(self, text=text,
                     font=ctk.CTkFont(size=theme.fs_heading(), weight="bold"),
                     text_color=theme.ACCENT()).pack(anchor="w", padx=14, pady=(14, 0))
        if sub:
            ctk.CTkLabel(self, text=sub, anchor="w", justify="left", wraplength=820,
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(anchor="w", padx=14, pady=(0, 4))

    def _build(self):
        self._heading(
            "Delivery Report",
            "Capture a Baseline and an Optimized configuration, then export a "
            "before/after report for your client. Metrics come from the Chat "
            "session and the Tune tab's best result.")

        # ── Capture controls ────────────────────────────────────────────────────
        cap = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=8)
        cap.pack(fill="x", padx=14, pady=(6, 4))
        inner = ctk.CTkFrame(cap, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(inner, text="Label",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        ctk.CTkOptionMenu(inner, variable=self.label_var, values=_LABELS,
                          width=130, height=30,
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(6, 14))
        ctk.CTkButton(inner, text="＋ Capture from Chat", width=170, height=30,
                      fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(), text_color=_BLACK,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      command=self._capture_chat).pack(side="left", padx=(0, 8))
        ctk.CTkButton(inner, text="＋ Capture from Tune", width=170, height=30,
                      fg_color=theme.ACCENT2(), hover_color=theme.ACCENT(), text_color=_BLACK,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      command=self._capture_tune).pack(side="left")

        # Client + notes
        meta = ctk.CTkFrame(self, fg_color="transparent")
        meta.pack(fill="x", padx=14, pady=(2, 4))
        ctk.CTkLabel(meta, text="Client", width=54, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        ctk.CTkEntry(meta, textvariable=self.client_var, width=200, height=30,
                     fg_color=theme.SURFACE2,
                     font=ctk.CTkFont(size=theme.fs_button()),
                     placeholder_text="(optional)").pack(side="left", padx=(6, 16))
        ctk.CTkLabel(meta, text="Notes", width=54, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.MUTED()).pack(side="left")
        ctk.CTkEntry(meta, textvariable=self.notes_var, height=30,
                     fg_color=theme.SURFACE2,
                     font=ctk.CTkFont(size=theme.fs_button()),
                     placeholder_text="One-line summary for the client").pack(
            side="left", fill="x", expand=True, padx=(6, 0))

        # ── Captured snapshots list (compact, single line per item) ─────────────
        self._heading("Captured Configurations")
        self.snaps_frame = ctk.CTkScrollableFrame(self, fg_color=theme.SURFACE, height=62)
        self.snaps_frame.pack(fill="x", padx=14, pady=(0, 6))

        # Bottom-anchored controls are packed FIRST so they are always reserved
        # and never squeezed off-screen when the preview grows. The preview is
        # packed last and fills the remaining middle space.

        # ── Status line (bottom) ─────────────────────────────────────────────────
        ctk.CTkLabel(self, textvariable=self.status_msg, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(side="bottom", fill="x",
                                                     padx=14, pady=(0, 8))

        # ── Actions (bottom) ─────────────────────────────────────────────────────
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="bottom", fill="x", padx=14, pady=(4, 2))
        ctk.CTkButton(bar, text="Copy", width=90, height=32,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._copy,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold")).pack(side="left")
        ctk.CTkButton(bar, text="Export to File…", width=140, height=32,
                      fg_color=theme.SUCCESS, hover_color="#1F8B4C", text_color="#DCE4EE",
                      command=self._export,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold")).pack(side="left", padx=(8, 0))
        ctk.CTkButton(bar, text="Clear All", width=100, height=32,
                      fg_color=theme.SURFACE2, hover_color=theme.DANGER,
                      command=self._clear,
                      font=ctk.CTkFont(size=theme.fs_button())).pack(side="right")

        # ── Preview (fills the middle) ───────────────────────────────────────────
        prow = ctk.CTkFrame(self, fg_color="transparent")
        prow.pack(fill="x", padx=14, pady=(8, 0))
        ctk.CTkLabel(prow, text="Report Preview",
                     font=ctk.CTkFont(size=theme.fs_heading(), weight="bold"),
                     text_color=theme.ACCENT()).pack(side="left")
        ctk.CTkOptionMenu(prow, variable=self.fmt_var, values=["Markdown", "Plain text"],
                          width=120, height=28, command=lambda _v: self._refresh_preview(),
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="right")

        self.preview = ctk.CTkTextbox(self, fg_color=theme.SURFACE2, height=160,
                                      font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
                                      text_color=theme.TEXT(), wrap="word")
        self.preview.pack(fill="both", expand=True, padx=14, pady=(2, 4))
        self.preview.configure(state="disabled")

        self._refresh_snaps()

    # ── Capture ──────────────────────────────────────────────────────────────────

    def _add_snapshot(self, snap: Snapshot | None, source: str):
        if snap is None:
            messagebox.showinfo(
                "Nothing to capture",
                f"No data available from {source}.\n\n"
                "Run the model and send at least one message (Chat) or complete a "
                "test (Tune) before capturing.",
                parent=self)
            return
        snap.label = self.label_var.get()
        # Replace any existing snapshot with the same label (re-capture).
        self.snapshots = [s for s in self.snapshots if s.label != snap.label]
        self.snapshots.append(snap)
        # Keep a stable, meaningful order matching the label list.
        order = {lbl: i for i, lbl in enumerate(_LABELS)}
        self.snapshots.sort(key=lambda s: order.get(s.label, 99))
        self._refresh_snaps()
        self._refresh_preview()
        self.status_msg.set(f"Captured '{snap.label}' from {source}.")
        # Auto-advance the label to encourage a before/after pair.
        if self.label_var.get() == _LABELS[0]:
            self.label_var.set(_LABELS[1])

    def _capture_chat(self):
        self._add_snapshot(self.capture_chat(), "Chat")

    def _capture_tune(self):
        self._add_snapshot(self.capture_tune(), "Tune")

    # ── Snapshot list ──────────────────────────────────────────────────────────

    def _refresh_snaps(self):
        for w in self.snaps_frame.winfo_children():
            w.destroy()
        if not self.snapshots:
            ctk.CTkLabel(self.snaps_frame, text="No configurations captured yet.",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(anchor="w", padx=8, pady=8)
            return
        for snap in self.snapshots:
            row = ctk.CTkFrame(self.snaps_frame, fg_color=theme.SURFACE2, corner_radius=6)
            row.pack(fill="x", padx=6, pady=2)
            ctk.CTkButton(row, text="✕", width=26, height=24,
                          fg_color=theme.SURFACE, hover_color=theme.DANGER,
                          command=lambda s=snap: self._remove(s),
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=4, pady=2)
            model = snap.model_name or "unknown"
            line = f"{snap.label} — {model}   ·   {self._snap_summary(snap)}"
            ctk.CTkLabel(row, text=line, anchor="w",
                         font=ctk.CTkFont(size=theme.fs_small(), weight="bold"),
                         text_color=theme.TEXT()).pack(side="left", fill="x", expand=True,
                                                       padx=8, pady=3)

    @staticmethod
    def _snap_summary(snap: Snapshot) -> str:
        parts = []
        if snap.metric("avg_tps") is not None:
            parts.append(f"{snap.metric('avg_tps')} tok/s")
        if snap.metric("avg_ttft") is not None:
            parts.append(f"TTFT {snap.metric('avg_ttft')}s")
        if snap.metric("score") is not None:
            parts.append(f"score {snap.metric('score')}")
        if snap.metric("total_tokens"):
            parts.append(f"{snap.metric('total_tokens')} tok")
        return "  ·  ".join(parts) if parts else "no metrics captured"

    def _remove(self, snap: Snapshot):
        self.snapshots = [s for s in self.snapshots if s is not snap]
        self._refresh_snaps()
        self._refresh_preview()

    # ── Preview / export ──────────────────────────────────────────────────────

    def _render(self) -> str:
        hw = hardware_summary()
        kwargs = dict(hardware=hw, session_notes=self.notes_var.get(),
                      client=self.client_var.get())
        if self.fmt_var.get() == "Plain text":
            return report.build_text(self.snapshots, **kwargs)
        return report.build_markdown(self.snapshots, **kwargs)

    def _refresh_preview(self):
        text = self._render()
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def _copy(self):
        if not self.snapshots:
            self.status_msg.set("Nothing to copy — capture a configuration first.")
            return
        self.clipboard_clear()
        self.clipboard_append(self._render())
        self.status_msg.set("Report copied to clipboard.")

    def _export(self):
        if not self.snapshots:
            messagebox.showinfo("Nothing to export",
                                "Capture at least one configuration first.", parent=self)
            return
        md = self.fmt_var.get() == "Markdown"
        ext = ".md" if md else ".txt"
        model = self.snapshots[-1].model_name or "model"
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=ext,
            initialfile=f"report_{safe}{ext}",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._render())
            self.status_msg.set(f"Report saved to {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self)

    def _clear(self):
        if not self.snapshots:
            return
        if not messagebox.askyesno("Clear all",
                                   "Remove all captured configurations?", parent=self):
            return
        self.snapshots = []
        self._refresh_snaps()
        self._refresh_preview()
        self.status_msg.set("Cleared.")

"""
ModelDownloadDialog — search Hugging Face for GGUF models and download them
into the configured models folder.

Network calls (search, file listing) run on worker threads and marshal results
back to the UI thread via .after(). The actual download is handled by
core.downloader.Downloader (also threaded, cancellable, with progress).
On a completed download the on_downloaded callback fires so the caller can
refresh its model list.
"""

from __future__ import annotations

import threading

import customtkinter as ctk

from . import theme
from ..core import downloader
from ..core.config import settings


class ModelDownloadDialog(ctk.CTkToplevel):
    def __init__(self, master, on_downloaded=None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)
        self._on_downloaded = on_downloaded or (lambda: None)

        self.title("Download Models — Hugging Face (GGUF)")
        self.geometry("880x600")
        self.minsize(720, 460)

        self.search_var = ctk.StringVar(value="")
        self.status_var = ctk.StringVar(value="Search for a model (e.g. 'qwen2.5 instruct', 'llama 3.1 8b').")
        self._busy = False

        self.dl = downloader.Downloader(
            on_progress=lambda d, t: self.after(0, self._on_dl_progress, d, t),
            on_done=lambda p: self.after(0, self._on_dl_done, p),
            on_error=lambda m: self.after(0, self._on_dl_error, m),
        )

        self._build()

        self.transient(master)
        try:
            self.grab_set()
        except Exception:
            pass
        self.after(50, self._center_on_master)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _e: self._close())

    # ── Layout ──────────────────────────────────────────────────────────────

    def _build(self):
        # Search row
        top = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0)
        top.pack(side="top", fill="x")
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=10)
        entry = ctk.CTkEntry(row, textvariable=self.search_var, height=34,
                             fg_color=theme.SURFACE2,
                             placeholder_text="Search Hugging Face GGUF models…",
                             font=ctk.CTkFont(size=theme.fs_base()))
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        entry.bind("<Return>", lambda _e: self._search())
        entry.focus_set()
        self.search_btn = ctk.CTkButton(row, text="Search", width=110, height=34,
                                        command=self._search,
                                        font=ctk.CTkFont(size=theme.fs_button(), weight="bold"))
        self.search_btn.pack(side="left")

        dest = settings.get_str("gguf_models_dir") or "(not set — configure in Settings)"
        ctk.CTkLabel(top, text=f"Downloads to:  {dest}", anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(fill="x", padx=12, pady=(0, 8))

        # Two panes: repos (left) and files (right)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True, padx=12, pady=(8, 4))
        body.grid_columnconfigure(0, weight=1, uniform="panes")
        body.grid_columnconfigure(1, weight=1, uniform="panes")
        body.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(body, text="Models", anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.ACCENT()).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(body, text="GGUF files", anchor="w",
                     font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                     text_color=theme.ACCENT()).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        self.repos_frame = ctk.CTkScrollableFrame(body, fg_color=theme.SURFACE)
        self.repos_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.files_frame = ctk.CTkScrollableFrame(body, fg_color=theme.SURFACE)
        self.files_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        # Footer: progress + status + cancel/close
        foot = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0)
        foot.pack(side="bottom", fill="x")
        self.progress = ctk.CTkProgressBar(foot, height=14, corner_radius=7,
                                           progress_color=theme.ACCENT(), fg_color=theme.SURFACE2)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=14, pady=(10, 2))
        srow = ctk.CTkFrame(foot, fg_color="transparent")
        srow.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(srow, textvariable=self.status_var, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(side="left", fill="x", expand=True)
        self.cancel_btn = ctk.CTkButton(srow, text="Cancel", width=90, height=30,
                                        fg_color=theme.SURFACE2, hover_color=theme.DANGER,
                                        command=self._cancel_download, state="disabled",
                                        font=ctk.CTkFont(size=theme.fs_small()))
        self.cancel_btn.pack(side="right", padx=(8, 0))
        ctk.CTkButton(srow, text="Close", width=90, height=30,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._close,
                      font=ctk.CTkFont(size=theme.fs_small())).pack(side="right")

    # ── Search ────────────────────────────────────────────────────────────────

    def _search(self):
        q = self.search_var.get().strip()
        if not q or self._busy:
            return
        self._set_busy(True)
        self.status_var.set(f"Searching for “{q}”…")
        self._clear(self.repos_frame)
        self._clear(self.files_frame)
        threading.Thread(target=self._search_worker, args=(q,), daemon=True).start()

    def _search_worker(self, q):
        try:
            hits = downloader.search_models(q)
            self.after(0, self._render_repos, hits)
        except Exception as e:
            self.after(0, self._search_failed, str(e))

    def _search_failed(self, msg):
        self._set_busy(False)
        self.status_var.set(f"Search failed: {msg}")

    def _render_repos(self, hits):
        self._set_busy(False)
        self._clear(self.repos_frame)
        if not hits:
            self.status_var.set("No GGUF models found. Try a different query.")
            return
        self.status_var.set(f"{len(hits)} model(s). Select one to see its GGUF files.")
        for h in hits:
            label = f"{h.repo_id}\n↓ {h.downloads:,}   ♥ {h.likes:,}"
            btn = ctk.CTkButton(self.repos_frame, text=label, anchor="w", height=44,
                                fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                                text_color=theme.TEXT(),
                                command=lambda r=h.repo_id: self._load_files(r),
                                font=ctk.CTkFont(size=theme.fs_small()))
            btn.pack(fill="x", padx=6, pady=3)

    # ── Files ──────────────────────────────────────────────────────────────────

    def _load_files(self, repo_id):
        if self._busy:
            return
        self._set_busy(True)
        self.status_var.set(f"Loading files for {repo_id}…")
        self._clear(self.files_frame)
        threading.Thread(target=self._files_worker, args=(repo_id,), daemon=True).start()

    def _files_worker(self, repo_id):
        try:
            files = downloader.list_gguf_files(repo_id)
            self.after(0, self._render_files, repo_id, files)
        except Exception as e:
            self.after(0, self._files_failed, str(e))

    def _files_failed(self, msg):
        self._set_busy(False)
        self.status_var.set(f"Could not list files: {msg}")

    def _render_files(self, repo_id, files):
        self._set_busy(False)
        self._clear(self.files_frame)
        if not files:
            self.status_var.set(f"No .gguf files found in {repo_id}.")
            return
        self.status_var.set(f"{len(files)} GGUF file(s) in {repo_id}. Pick a quantization to download.")
        for gf in files:
            rowf = ctk.CTkFrame(self.files_frame, fg_color=theme.SURFACE2, corner_radius=6)
            rowf.pack(fill="x", padx=6, pady=3)
            ctk.CTkLabel(rowf, text=f"{gf.filename}\n{gf.size_h}", anchor="w", justify="left",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.TEXT()).pack(side="left", fill="x", expand=True,
                                                       padx=(8, 4), pady=4)
            ctk.CTkButton(rowf, text="Download", width=96, height=30,
                          command=lambda g=gf: self._download(g),
                          font=ctk.CTkFont(size=theme.fs_small(), weight="bold")).pack(
                side="right", padx=6, pady=4)

    # ── Download ────────────────────────────────────────────────────────────────

    def _download(self, gf):
        dest = settings.get_str("gguf_models_dir")
        if not dest:
            self.status_var.set("Set a models folder in Settings first (Downloads to: …).")
            return
        if self.dl.is_running:
            self.status_var.set("A download is already in progress.")
            return
        self.progress.set(0)
        self.status_var.set(f"Downloading {gf.filename} …")
        self.cancel_btn.configure(state="normal")
        self.search_btn.configure(state="disabled")
        self._dl_name = gf.filename
        self.dl.start(gf.repo_id, gf.path, dest)

    def _cancel_download(self):
        if self.dl.is_running:
            self.dl.cancel()
            self.status_var.set("Cancelling…")

    def _on_dl_progress(self, done, total):
        if total:
            self.progress.set(done / total)
            self.status_var.set(
                f"Downloading {getattr(self, '_dl_name', '')} — "
                f"{downloader.human_size(done)} / {downloader.human_size(total)}")
        else:
            self.status_var.set(
                f"Downloading {getattr(self, '_dl_name', '')} — {downloader.human_size(done)}")

    def _on_dl_done(self, path):
        self.progress.set(1.0)
        self.cancel_btn.configure(state="disabled")
        self.search_btn.configure(state="normal")
        self.status_var.set(f"Saved: {path}")
        try:
            self._on_downloaded()
        except Exception:
            pass

    def _on_dl_error(self, msg):
        self.cancel_btn.configure(state="disabled")
        self.search_btn.configure(state="normal")
        self.status_var.set(msg if msg == "Cancelled." else f"Download failed: {msg}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.search_btn.configure(state="disabled" if busy else "normal")

    @staticmethod
    def _clear(frame):
        for w in frame.winfo_children():
            w.destroy()

    def _center_on_master(self):
        try:
            self.update_idletasks()
            m = self.master
            mx, my = m.winfo_rootx(), m.winfo_rooty()
            mw, mh = m.winfo_width(), m.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            self.geometry(f"+{max(0, mx + (mw - w) // 2)}+{max(0, my + (mh - h) // 2)}")
        except Exception:
            pass

    def _close(self):
        if self.dl.is_running:
            self.dl.cancel()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

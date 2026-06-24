"""
RagPanel — "Chat with your documents" (BM25, offline).

Add PDFs / text files, build a keyword (BM25) index, and toggle whether the
Chat tab augments each question with the most relevant excerpts. A search box
lets you preview what would be retrieved for a query.

Shares a single RagStore with the Chat tab so the toggle + index are live.
"""

from __future__ import annotations

import customtkinter as ctk
from tkinter import filedialog, messagebox

from . import theme
from ..core.rag import RagStore


_BLACK = "#10141C"


class RagPanel(ctk.CTkFrame):
    def __init__(self, master, store: RagStore | None = None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)
        self.store = store or RagStore()
        self.on_switch_to_chat = None  # callback set by app.py

        self.use_var = ctk.BooleanVar(value=self.store.enabled)
        self.topk_var = ctk.IntVar(value=self.store.top_k)
        self.status_msg = ctk.StringVar(value="Add documents, then enable 'Use in chat'.")
        self.search_var = ctk.StringVar(value="")

        self._build()
        self._refresh_docs()

    # ── Layout ──────────────────────────────────────────────────────────────────

    def _heading(self, text: str, sub: str = ""):
        ctk.CTkLabel(self, text=text,
                     font=ctk.CTkFont(size=theme.fs_heading(), weight="bold"),
                     text_color=theme.ACCENT()).pack(anchor="w", padx=14, pady=(14, 0))
        if sub:
            ctk.CTkLabel(self, text=sub, anchor="w",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(anchor="w", padx=14, pady=(0, 4))

    def _build(self):
        # ── Controls row ────────────────────────────────────────────────────────
        bar = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=8)
        bar.pack(fill="x", padx=14, pady=(12, 4))

        ctk.CTkButton(bar, text="+ Add documents", width=140, height=32,
                      fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(), text_color=_BLACK,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      command=self._add_docs).pack(side="left", padx=(10, 8), pady=8)
        ctk.CTkButton(bar, text="Clear all", width=90, height=32,
                      fg_color=theme.SURFACE2, hover_color=theme.BORDER,
                      command=self._clear_all,
                      font=ctk.CTkFont(size=theme.fs_button())).pack(side="left", padx=(0, 16))

        self.use_switch = ctk.CTkSwitch(
            bar, text="Use in chat", variable=self.use_var, command=self._toggle_use,
            font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
            progress_color=theme.SUCCESS)
        self.use_switch.pack(side="left", padx=(0, 16))

        ctk.CTkLabel(bar, text="Excerpts per question",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(side="left")
        ctk.CTkLabel(bar, text="(how many doc snippets to include)",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(side="left", padx=(4, 0))
        ctk.CTkOptionMenu(bar, width=70, height=30, variable=self.topk_var,
                          values=["2", "3", "4", "6", "8"], command=self._set_topk,
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="left", padx=(6, 10))

        # ── Documents list ────────────────────────────────────────────────────────
        self._heading("Documents", "PDF, TXT, or Markdown. Text is extracted and split into searchable chunks.")
        self.docs_frame = ctk.CTkScrollableFrame(self, fg_color=theme.SURFACE, height=80)
        self.docs_frame.pack(fill="x", padx=14, pady=(0, 6))

        # ── Test search ────────────────────────────────────────────────────────────
        self._heading("Test Search", "Try a question to preview which excerpts would be sent to the model.")
        srow = ctk.CTkFrame(self, fg_color="transparent")
        srow.pack(fill="x", padx=14, pady=(0, 4))
        ent = ctk.CTkEntry(srow, textvariable=self.search_var, height=32, fg_color=theme.SURFACE2,
                           font=ctk.CTkFont(size=theme.fs_small()),
                           placeholder_text="e.g. What is the refund policy?")
        ent.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ent.bind("<Return>", lambda _e: self._search())
        ctk.CTkButton(srow, text="Search", width=90, height=32,
                      fg_color=theme.ACCENT(), hover_color=theme.ACCENT2(), text_color=_BLACK,
                      font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                      command=self._search).pack(side="right")

        self.preview = ctk.CTkTextbox(self, fg_color=theme.SURFACE2, height=140,
                                      font=ctk.CTkFont(family="Consolas", size=theme.fs_code()),
                                      text_color=theme.TEXT(), wrap="word")
        self.preview.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self.preview.configure(state="disabled")

        ctk.CTkLabel(self, textvariable=self.status_msg, anchor="w",
                     font=ctk.CTkFont(size=theme.fs_small()),
                     text_color=theme.MUTED()).pack(fill="x", padx=14, pady=(0, 10))

    # ── Documents ──────────────────────────────────────────────────────────────

    def _add_docs(self):
        paths = filedialog.askopenfilenames(
            title="Add documents",
            filetypes=[("Documents", "*.pdf *.txt *.md *.markdown"), ("All files", "*.*")],
            parent=self)
        if not paths:
            return
        added_chunks = 0
        failed = []
        for p in paths:
            n = self.store.add_file(p)
            if n:
                added_chunks += n
            else:
                failed.append(p)
        self.store.build()
        self._refresh_docs()
        msg = f"Added {added_chunks} chunk(s) from {len(paths) - len(failed)} file(s)."
        if failed:
            msg += f"  Could not read {len(failed)} file(s) (no extractable text?)."
        self.status_msg.set(msg)
        # Offer to switch to Chat
        if added_chunks > 0 and self.on_switch_to_chat:
            self._show_go_to_chat()

    def _clear_all(self):
        if not self.store.has_docs:
            return
        if messagebox.askyesno("Clear documents", "Remove all documents from the index?", parent=self):
            self.store.clear()
            self._refresh_docs()
            self.status_msg.set("Cleared all documents.")

    def _remove_doc(self, name: str):
        self.store.remove_document(name)
        self.store.build()
        self._refresh_docs()
        self.status_msg.set(f"Removed '{name}'.")

    def _refresh_docs(self):
        for w in self.docs_frame.winfo_children():
            w.destroy()
        if not self.store.docs:
            ctk.CTkLabel(self.docs_frame, text="No documents added yet.",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(anchor="w", padx=10, pady=8)
            return
        for name, count in sorted(self.store.docs.items()):
            row = ctk.CTkFrame(self.docs_frame, fg_color=theme.SURFACE2, corner_radius=6)
            row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(row, text=name, anchor="w",
                         font=ctk.CTkFont(size=theme.fs_button(), weight="bold"),
                         text_color=theme.TEXT()).pack(side="left", padx=(10, 6), pady=5)
            ctk.CTkLabel(row, text=f"{count} chunk(s)", anchor="w",
                         font=ctk.CTkFont(size=theme.fs_small()),
                         text_color=theme.MUTED()).pack(side="left")
            ctk.CTkButton(row, text="Remove", width=70, height=26,
                          fg_color=theme.SURFACE, hover_color=theme.DANGER,
                          command=lambda n=name: self._remove_doc(n),
                          font=ctk.CTkFont(size=theme.fs_small())).pack(side="right", padx=6, pady=4)

    # ── Toggles ──────────────────────────────────────────────────────────────────

    def _show_go_to_chat(self):
        """Ask user if they want to add more docs or switch to Chat."""
        go = messagebox.askyesno(
            "Documents added",
            "Documents indexed successfully.\n\n"
            "Switch to Chat to try them?\n"
            "(You can also add more documents first.)",
            parent=self)
        if go and self.on_switch_to_chat:
            self.on_switch_to_chat()

    def _toggle_use(self):
        self.store.enabled = bool(self.use_var.get())
        if self.store.enabled and not self.store.has_docs:
            self.status_msg.set("Enabled, but no documents added yet — add some first.")
        else:
            self.status_msg.set("Chat will use your documents." if self.store.enabled
                                else "Chat will ignore documents.")

    def _set_topk(self, val):
        try:
            self.store.top_k = int(val)
        except (TypeError, ValueError):
            pass

    # ── Search preview ─────────────────────────────────────────────────────────

    def _search(self):
        query = self.search_var.get().strip()
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        if not query:
            self.preview.insert("1.0", "Type a question above and press Search.")
        elif not self.store.has_docs:
            self.preview.insert("1.0", "No documents indexed yet.")
        else:
            hits = self.store.retrieve(query)
            if not hits:
                self.preview.insert("1.0", "No matching excerpts found for that query.")
            else:
                lines = []
                for c, score in hits:
                    lines.append(f"[{c.label}]  (score {score:.2f})")
                    lines.append(c.text.strip())
                    lines.append("")
                self.preview.insert("1.0", "\n".join(lines))
                self.status_msg.set(f"{len(hits)} excerpt(s) would be sent as context.")
        self.preview.configure(state="disabled")

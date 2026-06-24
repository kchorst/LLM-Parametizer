"""
SettingsDialog — modal popup hosting the SettingsPanel.

Opens as its own Toplevel window so its close button (X) closes ONLY the
settings, never the main app. Save triggers the on_saved callback (used by the
app to refresh the chat model list) and then closes the dialog.
"""

from __future__ import annotations

import customtkinter as ctk

from . import theme
from .settings_panel import SettingsPanel


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master, on_saved=None, **kw):
        super().__init__(master, fg_color=theme.BG, **kw)
        self._external_on_saved = on_saved or (lambda: None)

        self.title("Settings")
        self.geometry("760x600")
        self.minsize(640, 460)

        # The panel calls our _handle_saved after a successful Save.
        self.panel = SettingsPanel(self, on_saved=self._handle_saved)
        self.panel.pack(fill="both", expand=True)

        # Modal-ish: stay above the main window and grab focus.
        self.transient(master)
        try:
            self.grab_set()
        except Exception:
            pass
        self.after(50, self._center_on_master)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _e: self._close())

    def _center_on_master(self):
        try:
            self.update_idletasks()
            m = self.master
            mx, my = m.winfo_rootx(), m.winfo_rooty()
            mw, mh = m.winfo_width(), m.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            x = mx + (mw - w) // 2
            y = my + (mh - h) // 2
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _handle_saved(self):
        # Notify the app (refresh chat models), then close the dialog.
        try:
            self._external_on_saved()
        finally:
            self._close()

    def _close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

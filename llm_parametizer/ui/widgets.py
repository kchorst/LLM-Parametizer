"""
Reusable UI widgets: StatusPill, MemoryMonitor.

These are thin CustomTkinter frames with simple update() methods so panels can
drive them from the UI thread.
"""

from __future__ import annotations

import customtkinter as ctk

from . import theme
from ..core import memory


class StatusPill(ctk.CTkFrame):
    """A prominent filled status badge whose background reflects backend state."""

    def __init__(self, master, label: str = "Stopped", **kw):
        super().__init__(master, fg_color=theme.SURFACE2, corner_radius=14, **kw)
        self._dot = ctk.CTkLabel(self, text="●", font=ctk.CTkFont(size=15),
                                 text_color=theme.MUTED())
        self._dot.pack(side="left", padx=(12, 5), pady=4)
        self._text = ctk.CTkLabel(
            self, text=label,
            font=ctk.CTkFont(size=theme.fs_base() + 2, weight="bold"),
            text_color=theme.MUTED(),
        )
        self._text.pack(side="left", padx=(0, 14), pady=4)

    def set(self, text: str, fg: str = theme.SURFACE2, text_color: str = "#FFFFFF",
            dot_color: str | None = None) -> None:
        self.configure(fg_color=fg)
        self._text.configure(text=text, text_color=text_color)
        self._dot.configure(text_color=dot_color or text_color)

    def set_state(self, state: str) -> None:
        # (badge background, text color, label)
        mapping = {
            "running":  (theme.SUCCESS, "#FFFFFF", "Running"),
            "ready":    (theme.SUCCESS, "#FFFFFF", "Running"),
            "starting": (theme.WARNING, "#1A1A1A", "Starting…"),
            "stopping": (theme.WARNING, "#1A1A1A", "Stopping…"),
            "stopped":  (theme.SURFACE2, theme.MUTED(), "Stopped"),
            "error":    (theme.DANGER, "#FFFFFF", "Error"),
        }
        fg, txt, label = mapping.get(state, (theme.SURFACE2, theme.MUTED(), state))
        self.set(label, fg=fg, text_color=txt)


class MetricsBar(ctk.CTkFrame):
    """A compact, subtle single-line metrics readout (TTFT / TPS / tokens / time).

    Renders nothing until real data arrives. The parent is responsible for
    showing/hiding the row based on `has_data` so no blank metrics are ever
    displayed before a model has run.
    """

    def __init__(self, master, title: str = "", **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._title = title
        self._has_data = False
        self._label = ctk.CTkLabel(
            self, text="", anchor="w",
            font=ctk.CTkFont(size=theme.fs_small()),
            text_color=theme.MUTED())
        self._label.pack(side="left", fill="x", expand=True)

    @staticmethod
    def _fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    def set_metrics(self, ttft=None, tps=None, tokens=None, elapsed=None) -> None:
        parts = []
        if ttft is not None:
            parts.append(f"TTFT {self._fmt(ttft)}s")
        if tps is not None:
            parts.append(f"{self._fmt(tps)} tok/s")
        if tokens is not None:
            parts.append(f"{self._fmt(tokens)} tok")
        if elapsed is not None:
            parts.append(f"{self._fmt(elapsed)}s")
        if not parts:
            self.clear()
            return
        text = "   ·   ".join(parts)
        if self._title:
            text = f"{self._title}  {text}"
        self._label.configure(text=text)
        self._has_data = True

    def clear(self) -> None:
        self._label.configure(text="")
        self._has_data = False

    @property
    def has_data(self) -> bool:
        return self._has_data


class MemoryMonitor(ctk.CTkFrame):
    """Displays system RAM (and VRAM if present) with a refresh poll."""

    def __init__(self, master, poll_ms: int = 3000, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._poll_ms = poll_ms

        self._ram = ctk.CTkLabel(self, text="RAM —",
                                 font=ctk.CTkFont(size=theme.fs_small()),
                                 text_color=theme.MUTED())
        self._ram.pack(side="left", padx=(0, 12))

        self._vram = ctk.CTkLabel(self, text="",
                                  font=ctk.CTkFont(size=theme.fs_small()),
                                  text_color=theme.MUTED())
        self._vram.pack(side="left")

        self._vram_checked = False
        self._vram_present = False
        self.refresh()
        self.after(self._poll_ms, self._tick)

    def _tick(self):
        self.refresh()
        try:
            self.after(self._poll_ms, self._tick)
        except Exception:
            pass

    def refresh(self):
        total, avail = memory.system_memory()
        if total:
            used = total - avail
            pct = int(used * 100 / total) if total else 0
            self._ram.configure(
                text=f"RAM {used/1024**3:.1f}/{total/1024**3:.1f} GB ({pct}%)",
                text_color=theme.DANGER if pct >= 90 else
                           (theme.WARNING if pct >= 75 else theme.MUTED()),
            )
        else:
            self._ram.configure(text="RAM —")

        # VRAM: only probe once to avoid repeated nvidia-smi cost if absent
        if not self._vram_checked:
            info = memory.vram_info()
            self._vram_checked = True
            self._vram_present = info.available
            if info.available:
                self._update_vram(info)
        elif self._vram_present:
            self._update_vram(memory.vram_info())

    def _update_vram(self, info):
        if info.available and info.total_bytes:
            pct = int(info.used_bytes * 100 / info.total_bytes)
            self._vram.configure(
                text=f"VRAM {info.used_bytes/1024**3:.1f}/{info.total_bytes/1024**3:.1f} GB ({pct}%)",
                text_color=theme.MUTED(),
            )

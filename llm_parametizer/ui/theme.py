"""
Theme: colors and fonts derived from settings.

Centralises the palette so panels stay visually consistent. Values read from the
Settings singleton with sensible dark-theme defaults.
"""

from __future__ import annotations

from ..core.config import settings

# ── Palette ───────────────────────────────────────────────────────────────────

BG = "#0B0F1A"
SURFACE = "#121826"
SURFACE2 = "#1A2234"
BORDER = "#2A3450"

SUCCESS = "#27AE60"
WARNING = "#F0A500"
DANGER = "#C0392B"
INFO = "#4F8EF7"


def TEXT() -> str:
    return settings.get_str("text_color", "#FFFFFF")


def MUTED() -> str:
    return settings.get_str("muted_color", "#8892B0")


def ACCENT() -> str:
    return settings.get_str("accent_color", "#4F8EF7")


def ACCENT2() -> str:
    return settings.get_str("accent2_color", "#A78BFA")


# ── Fonts ───────────────────────────────────────────────────────────────────────

def fs_base() -> int:
    return settings.get_int("font_size_base", 13)


def fs_code() -> int:
    return settings.get_int("font_size_code", 13)


def fs_small() -> int:
    return settings.get_int("font_size_small", 11)


# ── Semantic sizes (use these instead of fs_base()+N ad-hoc) ─────────────────

def fs_label() -> int:
    """Field labels and secondary text."""
    return fs_small()


def fs_button() -> int:
    """Button labels (larger than secondary text for legibility)."""
    return fs_base() + 1


def fs_heading() -> int:
    """Section headings within a panel."""
    return fs_base() + 2


def fs_title() -> int:
    """Dialog / modal titles."""
    return fs_base() + 5

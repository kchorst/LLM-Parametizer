"""
Path resolution for binaries and GGUF model files.

Resolves llama.cpp binaries (server/cli/swap) from explicit paths, a configured
bin directory, or the system PATH. Scans the models directory for .gguf files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


# ── Binary resolution ───────────────────────────────────────────────────────────

def _exe(name: str) -> str:
    """Append .exe on Windows if not already present."""
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return name + ".exe"
    return name


def resolve_binary(explicit_path: str, bin_dir: str, binary_name: str) -> str:
    """
    Resolve a binary in priority order:
      1. explicit_path (if it points to an existing file)
      2. bin_dir / binary_name (with .exe on Windows)
      3. system PATH lookup
    Returns the resolved path, or "" if not found.
    """
    # 1. Explicit full path
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p)

    # 2. Bin directory
    if bin_dir:
        candidate = Path(bin_dir) / _exe(binary_name)
        if candidate.is_file():
            return str(candidate)
        # Also try without forced .exe (cross-platform safety)
        candidate2 = Path(bin_dir) / binary_name
        if candidate2.is_file():
            return str(candidate2)

    # 3. System PATH
    found = shutil.which(binary_name) or shutil.which(_exe(binary_name))
    if found:
        return found

    return ""


def resolve_llama_server(explicit: str = "", bin_dir: str = "") -> str:
    return resolve_binary(explicit, bin_dir, "llama-server")


def resolve_llama_cli(explicit: str = "", bin_dir: str = "") -> str:
    return resolve_binary(explicit, bin_dir, "llama-cli")


def resolve_llama_swap(explicit: str = "", bin_dir: str = "") -> str:
    return resolve_binary(explicit, bin_dir, "llama-swap")


def binary_exists(path: str) -> bool:
    """True if the path resolves to an existing file or a PATH entry."""
    if not path:
        return False
    if Path(path).is_file():
        return True
    return bool(shutil.which(path))


# ── GGUF discovery ───────────────────────────────────────────────────────────────

class GGUFEntry:
    """A discovered GGUF model file."""

    __slots__ = ("full_path", "display_name", "size_bytes")

    def __init__(self, full_path: str, display_name: str, size_bytes: int):
        self.full_path = full_path
        self.display_name = display_name
        self.size_bytes = size_bytes

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024 ** 3

    @property
    def stem(self) -> str:
        return Path(self.full_path).stem

    def __repr__(self) -> str:
        return f"GGUFEntry({self.display_name!r}, {self.size_gb:.2f} GB)"


def scan_gguf_models(models_dir: str) -> list[GGUFEntry]:
    """
    Recursively scan a directory for *.gguf files.
    Returns a list of GGUFEntry sorted by display name. Empty list on any error.
    """
    if not models_dir:
        return []
    root = Path(models_dir)
    if not root.is_dir():
        return []

    entries: list[GGUFEntry] = []
    try:
        for f in sorted(root.rglob("*.gguf"), key=lambda p: str(p).lower()):
            try:
                size = f.stat().st_size
            except Exception:
                size = 0
            try:
                display = str(f.relative_to(root))
            except ValueError:
                display = f.name
            entries.append(GGUFEntry(str(f), display, size))
    except Exception:
        return entries
    return entries


def gguf_path_for_display(models_dir: str, display_name: str) -> str:
    """Map a display name back to its full path. Returns absolute path or ''."""
    if not display_name:
        return ""
    # If already absolute and exists, return as-is
    p = Path(display_name)
    if p.is_absolute() and p.is_file():
        return str(p)
    if models_dir:
        candidate = Path(models_dir) / display_name
        if candidate.is_file():
            return str(candidate)
    return ""

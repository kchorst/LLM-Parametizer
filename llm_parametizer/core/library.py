"""
Profile library — named INI profiles on disk.

Stores ModelConfig profiles as .ini files under config.LIBRARY_DIR
(~/.llm_parametizer/library). Profile names map to safe filenames. All IO is
best-effort: methods return bool/None on failure and never raise.

Practical "version history" is achieved by saving named profiles
(e.g. "qwen-v1", "qwen-v2"). A deep version tree is out of scope for Phase 4.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import config
from ..backend import llama_cpp as lc
from .models import ModelConfig


_SAFE = re.compile(r"[^A-Za-z0-9 ._-]+")


def sanitize_name(name: str) -> str:
    """Turn a profile name into a safe base filename (no extension)."""
    name = (name or "").strip()
    name = _SAFE.sub("_", name)
    name = name.strip(" .")
    return name or "untitled"


class ProfileLibrary:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory) if directory else config.LIBRARY_DIR
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ── Paths ───────────────────────────────────────────────────────────────────

    def path_for(self, name: str) -> Path:
        return self.dir / f"{sanitize_name(name)}.ini"

    def exists(self, name: str) -> bool:
        return self.path_for(name).is_file()

    # ── Listing ──────────────────────────────────────────────────────────────

    def list_profiles(self) -> list[str]:
        """Sorted list of profile names (filename stems), .ini only."""
        try:
            return sorted(p.stem for p in self.dir.glob("*.ini") if p.is_file())
        except Exception:
            return []

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def save(self, name: str, cfg: ModelConfig) -> bool:
        """Write cfg as an INI profile. Stamps config_name with the saved name."""
        try:
            cfg.config_name = sanitize_name(name)
            text = lc.to_ini(cfg)
            self.path_for(name).write_text(text, encoding="utf-8")
            return True
        except Exception:
            return False

    def load(self, name: str) -> ModelConfig | None:
        """Read a profile into a ModelConfig, or None if missing/unreadable."""
        path = self.path_for(name)
        try:
            if not path.is_file():
                return None
            return lc.from_ini(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete(self, name: str) -> bool:
        try:
            path = self.path_for(name)
            if path.is_file():
                path.unlink()
                return True
            return False
        except Exception:
            return False

    def rename(self, old: str, new: str) -> bool:
        try:
            src = self.path_for(old)
            dst = self.path_for(new)
            if not src.is_file() or dst.exists():
                return False
            # Update config_name inside the file, then move.
            cfg = self.load(old)
            if cfg is None:
                return False
            cfg.config_name = sanitize_name(new)
            dst.write_text(lc.to_ini(cfg), encoding="utf-8")
            src.unlink()
            return True
        except Exception:
            return False

    def duplicate(self, name: str, new: str) -> bool:
        try:
            cfg = self.load(name)
            if cfg is None or self.exists(new):
                return False
            return self.save(new, cfg)
        except Exception:
            return False

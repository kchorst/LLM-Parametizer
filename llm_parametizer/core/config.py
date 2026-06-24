"""
Settings persistence for LLM Parametizer.

Loads/saves a single JSON settings file under the user's home directory.
Provides typed getters with defaults and a singleton Settings instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Storage location ───────────────────────────────────────────────────────────

APP_DIR = Path.home() / ".llm_parametizer"
LEGACY_APP_DIR = Path.home() / ".modelforge"   # pre-rename location (migrated on first run)
SETTINGS_FILE = APP_DIR / "settings.json"
LIBRARY_DIR = APP_DIR / "library"      # saved INI profiles + modelfiles
SESSIONS_DIR = APP_DIR / "sessions"    # chat session storage

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_SETTINGS: dict[str, Any] = {
    # UI
    "font_size_base": 13,
    "font_size_code": 13,
    "font_size_small": 11,
    "tab_font_size": 15,
    "ui_scale": 1.0,
    "window_geometry": "",   # 'WxH+X+Y' restored on launch; saved on close
    "appearance_mode": "dark",
    "color_theme": "dark-blue",
    "text_color": "#FFFFFF",
    "muted_color": "#8892B0",
    "accent_color": "#4F8EF7",
    "accent2_color": "#A78BFA",
    "left_panel_width": 390,
    "right_panel_width": 320,

    # Backend — llama.cpp is PRIMARY
    "default_backend": "llama.cpp (server)",
    "llama_bin_dir": "",          # directory containing llama-server / llama-cli
    "llama_server_path": "",      # optional override (full path)
    "llama_cli_path": "",
    "llama_swap_path": "",
    "gguf_models_dir": "",        # where .gguf files live and downloads land
    "llama_host": "127.0.0.1",
    "llama_port": 8080,
    "swap_listen": "127.0.0.1:8090",
    "n_gpu_layers": 0,
    "threads": -1,
    "flash_attn": False,
    "cont_batching": True,
    "n_ctx": 4096,               # default context window size

    # Ollama — SECONDARY
    "ollama_host": "127.0.0.1",
    "ollama_port": 11434,

    # Memory
    "ram_headroom_gb": 1.5,       # reserve at least this for OS/app
    "ram_headroom_pct": 0.12,     # or this fraction of total, whichever is larger
}


class Settings:
    """Thread-agnostic settings container backed by a JSON file."""

    def __init__(self, path: Path = SETTINGS_FILE):
        self._path = path
        self._data: dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.load()

    # ── Dict-like access ──────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key, DEFAULT_SETTINGS.get(key))

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self._data[key]
        if default is not None:
            return default
        return DEFAULT_SETTINGS.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        self._data.update(values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    # ── Typed getters ───────────────────────────────────────────────────────────

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val)

    def get_str(self, key: str, default: str = "") -> str:
        val = self.get(key, default)
        return "" if val is None else str(val)

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load settings from disk, merging over defaults. Never raises."""
        try:
            if self._path.is_file():
                with open(self._path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = dict(DEFAULT_SETTINGS)
                    self._data.update(loaded)
        except Exception:
            self._data = dict(DEFAULT_SETTINGS)

    def save(self) -> bool:
        """Persist settings to disk. Returns True on success."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        """Reset all settings to defaults (in memory; call save() to persist)."""
        self._data = dict(DEFAULT_SETTINGS)


def migrate_legacy_dir() -> bool:
    """
    One-time migration: if the pre-rename data dir (~/.modelforge) exists and the
    new dir (~/.llm_parametizer) does not, rename it so saved profiles, settings,
    session notes, and the RAG cache carry over. Best-effort; never raises.
    Returns True if a migration was performed.
    """
    try:
        if LEGACY_APP_DIR.is_dir() and not APP_DIR.exists():
            LEGACY_APP_DIR.rename(APP_DIR)
            return True
    except Exception:
        pass
    return False


def ensure_dirs() -> None:
    """Create the application directories if they do not exist."""
    for d in (APP_DIR, LIBRARY_DIR, SESSIONS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


# Singleton instance used throughout the app.
migrate_legacy_dir()
ensure_dirs()
settings = Settings()

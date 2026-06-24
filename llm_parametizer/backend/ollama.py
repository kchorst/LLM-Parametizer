"""
Ollama API client (SECONDARY backend).

Thin wrapper over the Ollama REST API for availability checks, model listing,
and chat. Streaming chat lives in chat/engine.py; this module provides the
synchronous helpers and the base-URL builder.
"""

from __future__ import annotations

import requests


def base_url(host: str = "127.0.0.1", port: int = 11434) -> str:
    return f"http://{host}:{port}"


def is_available(host: str = "127.0.0.1", port: int = 11434, timeout: float = 2.0) -> bool:
    """True if the Ollama daemon responds on /api/tags."""
    try:
        r = requests.get(f"{base_url(host, port)}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_models(host: str = "127.0.0.1", port: int = 11434) -> list[str]:
    """Return a list of installed model names. Empty on error."""
    try:
        r = requests.get(f"{base_url(host, port)}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", []) or []
        return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        return []


def model_info(name: str, host: str = "127.0.0.1", port: int = 11434) -> dict:
    """Return /api/show metadata for a model. Empty dict on error."""
    try:
        r = requests.post(f"{base_url(host, port)}/api/show", json={"name": name}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def running_models(host: str = "127.0.0.1", port: int = 11434) -> list[str]:
    """Return names of currently loaded models via /api/ps. Empty on error."""
    try:
        r = requests.get(f"{base_url(host, port)}/api/ps", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in (r.json().get("models", []) or []) if m.get("name")]
    except Exception:
        return []

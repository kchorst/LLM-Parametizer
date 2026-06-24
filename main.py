"""
LLM Parametizer entry point.

    python main.py
"""

from __future__ import annotations

import atexit

from llm_parametizer.ui.app import run
from llm_parametizer.core import ipc


def _cleanup():
    """Last-resort cleanup: stop IPC and kill any orphaned llama processes."""
    try:
        ipc.stop()
    except Exception:
        pass
    try:
        from llm_parametizer.backend.process import kill_orphans
        for name in ("llama-server", "llama-cli", "llama-swap"):
            kill_orphans(name)
    except Exception:
        pass


if __name__ == "__main__":
    atexit.register(_cleanup)
    ipc.start()
    run()

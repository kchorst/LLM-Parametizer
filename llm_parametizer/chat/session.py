"""
ChatSession — orchestrates history + streaming engine on a background thread.

The UI subscribes via callbacks:
  on_start()                 — generation began
  on_token(text)             — a content chunk arrived (accumulate in UI)
  on_done(full_text, stats)  — generation finished successfully
  on_error(message)          — generation failed

send() is non-blocking: it appends the user message, then streams the assistant
reply on a daemon thread, invoking callbacks. Exactly one generation runs at a
time; concurrent send() calls while busy are ignored.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from .engine import Backend, ChatStats, stream_chat
from .history import ChatHistory


class ChatSession:
    def __init__(
        self,
        history: Optional[ChatHistory] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str, ChatStats], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.history = history or ChatHistory()
        self.on_start = on_start or (lambda: None)
        self.on_token = on_token or (lambda _t: None)
        self.on_done = on_done or (lambda _f, _s: None)
        self.on_error = on_error or (lambda _m: None)

        self._busy = threading.Lock()
        self._cancel = threading.Event()

    @property
    def is_busy(self) -> bool:
        return self._busy.locked()

    def cancel(self) -> None:
        self._cancel.set()

    def clear(self) -> None:
        self.history.clear()

    def send(
        self,
        prompt: str,
        backend: Backend,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        model: str = "",
        system_prompt: str = "",
        num_ctx: int = 4096,
        options: dict | None = None,
    ) -> bool:
        """
        Append the user prompt and stream the assistant reply on a thread.
        Returns False immediately if a generation is already in progress or the
        prompt is empty; True if generation was started.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return False
        if not self._busy.acquire(blocking=False):
            return False

        self._cancel.clear()
        self.history.add_user(prompt)
        payload = self.history.build_payload(system_prompt=system_prompt, num_ctx=num_ctx)

        def run():
            full = ""
            final: ChatStats | None = None
            try:
                self.on_start()
                for item in stream_chat(
                    backend, payload,
                    host=host, port=port, model=model, options=options,
                ):
                    if self._cancel.is_set():
                        break
                    if isinstance(item, ChatStats):
                        final = item
                        break
                    full += item
                    self.on_token(item)

                if final and final.error:
                    self.on_error(final.error)
                else:
                    self.history.add_assistant(full)
                    self.on_done(full, final or ChatStats(tokens=len(full.split())))
            except Exception as e:
                self.on_error(str(e))
            finally:
                self._busy.release()

        threading.Thread(target=run, daemon=True).start()
        return True

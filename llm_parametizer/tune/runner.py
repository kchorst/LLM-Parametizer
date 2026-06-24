"""
TuneRunner — executes a SweepSpec against a running backend on a worker thread.

For each parameter combo, every prompt in the suite is sent (non-streaming use
of the streaming engine: chunks are concatenated), scored, and aggregated into
a ComboScore. Progress and per-combo results are reported via callbacks, which
the UI marshals onto the main thread. The run is cancellable at combo and
prompt boundaries.

The backend (llama-server) must already be running; the runner does NOT start
or restart it. num_ctx is therefore held fixed (see tuning.SWEEPABLE).
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from ..chat.engine import Backend, ChatStats, stream_chat
from ..core import tuning
from ..core.tuning import ComboScore, RunResult, SweepSpec


class TuneRunner:
    def __init__(
        self,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_combo: Optional[Callable[[ComboScore], None]] = None,
        on_finished: Optional[Callable[[list], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.on_progress = on_progress or (lambda d, t, label: None)
        self.on_combo = on_combo or (lambda c: None)
        self.on_finished = on_finished or (lambda results: None)
        self.on_error = on_error or (lambda m: None)

        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._rag_store = None   # set per-run via start(); used when spec.use_rag

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()

    def start(
        self,
        spec: SweepSpec,
        backend: Backend,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        model: str = "",
        timeout: float = 120.0,
        cap: int = tuning.MAX_COMBOS,
        rag_store=None,
    ) -> bool:
        """Begin a sweep on a daemon thread. Returns False if already running.

        When spec.use_rag is True and a rag_store is provided, each prompt's
        retrieved document context is injected into that prompt's system message
        (mirroring the Chat path) so the sweep scores document-grounded answers.
        """
        if self.is_running:
            return False
        self._rag_store = rag_store
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(spec, backend, host, port, model, timeout, cap),
            daemon=True,
        )
        self._thread.start()
        return True

    # ── Worker ─────────────────────────────────────────────────────────────────

    def _run(self, spec, backend, host, port, model, timeout, cap):
        try:
            combos = tuning.expand_grid(spec, cap=cap)
            prompts = spec.prompts or list(tuning.DEFAULT_PROMPTS)
            total = len(combos) * len(prompts)
            done = 0
            scored: list[ComboScore] = []

            for ci, params in enumerate(combos):
                if self._cancel.is_set():
                    break
                runs: list[RunResult] = []
                for prompt in prompts:
                    if self._cancel.is_set():
                        break
                    label = f"combo {ci + 1}/{len(combos)} — {self._swept(params)}"
                    self.on_progress(done, total, label)

                    run = self._run_one(ci, params, prompt, spec, backend, host, port, model, timeout)
                    runs.append(run)
                    done += 1
                    self.on_progress(done, total, label)

                combo = tuning.aggregate(ci, params, runs)
                scored.append(combo)
                self.on_combo(combo)

            self.on_finished(tuning.rank(scored))
        except Exception as e:
            self.on_error(str(e))

    def _run_one(self, ci, params, prompt, spec, backend, host, port, model, timeout) -> RunResult:
        system = spec.system_prompt
        if spec.use_rag and self._rag_store is not None:
            try:
                context, _chunks = self._rag_store.context_for(prompt)
                if context:
                    system = f"{system}\n\n{context}" if system else context
            except Exception:
                pass   # retrieval is best-effort; never fail a run on RAG errors

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        full = ""
        final: ChatStats | None = None
        try:
            for item in stream_chat(backend, messages, host=host, port=port,
                                    model=model, options=params, timeout=timeout):
                if self._cancel.is_set():
                    break
                if isinstance(item, ChatStats):
                    final = item
                    break
                full += item
        except Exception as e:
            return RunResult(combo_index=ci, prompt=prompt, error=str(e))

        if final and final.error:
            return RunResult(combo_index=ci, prompt=prompt, error=final.error)

        tps = final.tokens_per_second if final else None
        ttft = final.ttft if final else None
        score, metrics = tuning.score_response(
            full, tps, goal=spec.goal, keyword=spec.keyword, model_hint=spec.model_hint,
            ttft=ttft,
        )
        return RunResult(combo_index=ci, prompt=prompt, output=full,
                         score=score, metrics=metrics)

    @staticmethod
    def _swept(params: dict) -> str:
        return " ".join(f"{k}={params[k]}" for k in tuning.SWEEPABLE if k in params)

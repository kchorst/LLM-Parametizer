"""
Shared application state.

ActiveConfig holds the parameters and prompts currently being edited in the
Models tab and consumed by the Chat tab. Panels subscribe to change
notifications so edits in one place propagate to others. The GGUF model
selection itself lives in the Chat top bar; ActiveConfig holds everything else
needed to build a ModelConfig for preview/export/run.
"""

from __future__ import annotations

from typing import Any, Callable

from .models import ModelConfig, PARAM_KEYS, default_params, coerce_param


class ActiveConfig:
    def __init__(self) -> None:
        self.params: dict[str, Any] = default_params()
        self.system_prompt: str = ""
        self.template: str = ""
        self.profile_name: str = ""
        self._observers: list[Callable[[], None]] = []

    # ── Observers ───────────────────────────────────────────────────────────────

    def subscribe(self, callback: Callable[[], None]) -> None:
        if callback not in self._observers:
            self._observers.append(callback)

    def notify(self) -> None:
        for cb in list(self._observers):
            try:
                cb()
            except Exception:
                pass

    # ── Mutators ─────────────────────────────────────────────────────────────

    def set_param(self, key: str, value: Any, notify: bool = True) -> None:
        if key in PARAM_KEYS:
            self.params[key] = coerce_param(key, value)
            if notify:
                self.notify()

    def set_params(self, params: dict, notify: bool = True) -> None:
        for k, v in (params or {}).items():
            if k in PARAM_KEYS:
                self.params[k] = coerce_param(k, v)
        if notify:
            self.notify()

    def set_system_prompt(self, text: str, notify: bool = True) -> None:
        self.system_prompt = text or ""
        if notify:
            self.notify()

    def set_template(self, text: str, notify: bool = True) -> None:
        self.template = text or ""
        if notify:
            self.notify()

    def reset_params(self, notify: bool = True) -> None:
        self.params = default_params()
        if notify:
            self.notify()

    def load_from(self, cfg: ModelConfig, notify: bool = True) -> None:
        """Adopt parameters/prompts/name from a ModelConfig (e.g. loaded profile)."""
        self.set_params(cfg.params(), notify=False)
        self.system_prompt = cfg.system_prompt or ""
        self.template = cfg.template or ""
        self.profile_name = cfg.config_name or ""
        if notify:
            self.notify()

    # ── Construction ─────────────────────────────────────────────────────────

    def to_model_config(self, gguf_path: str = "", **backend_kwargs) -> ModelConfig:
        """
        Build a ModelConfig from the active params/prompts plus the given GGUF
        path and backend fields (host/port/paths/etc. supplied by the caller).
        """
        return ModelConfig.from_params(
            self.params,
            gguf_path=gguf_path,
            system_prompt=self.system_prompt,
            template=self.template,
            config_name=self.profile_name,
            **backend_kwargs,
        )

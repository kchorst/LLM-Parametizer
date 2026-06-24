"""
ModelConfig and parameter definitions for LLM Parametizer.

ModelConfig is the single shared configuration object. Adapters in the backend
package translate it into llama.cpp argv, llama-swap config, or Ollama Modelfile.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ── Parameter catalogue ────────────────────────────────────────────────────────
# Single source of truth for sampling parameters: default, range, type, label.

PARAMETERS: dict[str, dict[str, Any]] = {
    "temperature":    {"default": 0.8, "min": 0.0, "max": 2.0,  "step": 0.05, "type": float, "label": "Temperature",    "desc": "Higher = more creative/random"},
    "top_p":          {"default": 0.9, "min": 0.0, "max": 1.0,  "step": 0.05, "type": float, "label": "Top P",          "desc": "Nucleus sampling cutoff"},
    "top_k":          {"default": 40,  "min": 0,   "max": 100,  "step": 1,    "type": int,   "label": "Top K",          "desc": "Limit to K likeliest tokens"},
    "repeat_penalty": {"default": 1.1, "min": 1.0, "max": 2.0,  "step": 0.05, "type": float, "label": "Repeat Penalty", "desc": "Higher = less repetition"},
    "num_ctx":        {"default": 4096,"min": 512, "max": 131072,"step": 512, "type": int,   "label": "Context Size",   "desc": "Token memory window"},
    "num_predict":    {"default": -1,  "min": -2,  "max": 8192, "step": 64,   "type": int,   "label": "Max Tokens",     "desc": "Reply length cap (-1 = no limit)"},
    "mirostat":       {"default": 0,   "min": 0,   "max": 2,    "step": 1,    "type": int,   "label": "Mirostat",       "desc": "0 off; 1/2 auto-tune perplexity"},
    "mirostat_tau":   {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.5,  "type": float, "label": "Mirostat Tau",   "desc": "Target perplexity (if on)"},
    "seed":           {"default": 0,   "min": 0,   "max": 999999,"step": 1,   "type": int,   "label": "Seed",           "desc": "Fixed seed = repeatable (0 = random)"},
}

PARAM_KEYS = tuple(PARAMETERS.keys())


def default_params() -> dict[str, Any]:
    """Return a fresh dict of all parameters set to their defaults."""
    return {k: v["default"] for k, v in PARAMETERS.items()}


def coerce_param(key: str, value: Any) -> Any:
    """Coerce a value to the correct type for the given parameter key."""
    spec = PARAMETERS.get(key)
    if not spec:
        return value
    caster = spec["type"]
    try:
        return caster(value)
    except (TypeError, ValueError):
        return spec["default"]


# ── ModelConfig ────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Shared configuration for any backend."""

    # Common
    system_prompt: str = ""
    template: str = ""

    # Sampling parameters
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 4096
    num_predict: int = -1
    mirostat: int = 0
    mirostat_tau: float = 5.0
    seed: int = 0

    # llama.cpp (PRIMARY)
    gguf_path: str = ""
    llama_mode: str = "server"        # "server" | "cli"
    llama_server_path: str = ""
    llama_cli_path: str = ""
    host: str = "127.0.0.1"
    port: int = 8080
    n_gpu_layers: int = 0             # 0 = CPU; -1 = all layers on GPU
    threads: int = -1                 # -1 = auto
    flash_attn: bool = False
    cont_batching: bool = True
    extra_llama_args: list[str] = field(default_factory=list)

    # llama-swap
    llama_swap_path: str = ""
    swap_listen: str = "127.0.0.1:8090"
    swap_models: list[dict] = field(default_factory=list)

    # Ollama (SECONDARY)
    ollama_base_model: str = ""

    # Meta
    config_name: str = ""
    tag: str = ""

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_params(cls, params: dict, **kwargs) -> "ModelConfig":
        """Construct from a parameters dict, with extra fields as kwargs."""
        cfg = cls(**kwargs)
        for key in PARAM_KEYS:
            if key in params:
                setattr(cfg, key, coerce_param(key, params[key]))
        return cfg

    def params(self) -> dict[str, Any]:
        """Return just the sampling parameters as a dict."""
        return {k: getattr(self, k) for k in PARAM_KEYS}

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Build a ModelConfig from a dict, ignoring unknown keys."""
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in (data or {}).items() if k in valid}
        return cls(**clean)

    @property
    def model_name(self) -> str:
        """Human-friendly model name derived from the GGUF path."""
        if self.gguf_path:
            return Path(self.gguf_path).stem
        if self.ollama_base_model:
            return self.ollama_base_model
        return "(no model)"

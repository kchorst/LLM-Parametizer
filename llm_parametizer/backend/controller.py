"""
BackendController — high-level backend orchestration for the UI.

Bridges Settings + ModelConfig + preflight + ProcessManager and exposes a small,
explicit API the UI can call. Decides the chat target (backend/host/port/model)
so the chat layer never has to guess (fixes legacy routing ambiguity).

Engine modes (from settings 'default_backend'):
    "llama.cpp (server)" → llama-server  (PRIMARY, has a port → chat target)
    "llama.cpp (cli)"     → llama-cli     (one-shot; not a chat target)
    "llama-swap"          → llama-swap    (port at swap_listen → chat target)
    "Ollama"              → external      (SECONDARY; no process to start)
"""

from __future__ import annotations

import collections
from typing import Callable, Optional

from ..core.config import Settings
from ..core.models import ModelConfig, default_params
from ..core import paths, memory
from ..chat.engine import Backend
from . import llama_cpp as lc
from .process import ProcessManager, kill_orphans, port_open


# Engine identifiers used in settings / UI dropdown
ENGINE_LLAMA_SERVER = "llama.cpp (server)"
ENGINE_LLAMA_CLI = "llama.cpp (cli)"
ENGINE_LLAMA_SWAP = "llama-swap"
ENGINE_OLLAMA = "Ollama"

ENGINES = (ENGINE_LLAMA_SERVER, ENGINE_LLAMA_CLI, ENGINE_LLAMA_SWAP, ENGINE_OLLAMA)


class BackendController:
    def __init__(
        self,
        settings: Settings,
        on_status: Optional[Callable[[str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ):
        self.settings = settings
        self.engine = settings.get_str("default_backend", ENGINE_LLAMA_SERVER)
        # Retain the full backend log in a bounded ring buffer for the log viewer,
        # while still forwarding each line to the UI's on_log callback.
        self._log_buffer: collections.deque[str] = collections.deque(maxlen=5000)
        self._user_on_log = on_log or (lambda _l: None)
        self.pm = ProcessManager(on_log=self._capture_log, on_status=on_status, on_ready=on_ready)
        self._params: dict = dict(default_params())
        self._system_prompt: str = ""
        self._gguf_path: str = ""

    # ── Log capture ────────────────────────────────────────────────────────────

    def _capture_log(self, line: str) -> None:
        self._log_buffer.append(line)
        try:
            self._user_on_log(line)
        except Exception:
            pass

    def log_text(self) -> str:
        """Full retained backend log as a single string."""
        return "\n".join(self._log_buffer)

    def log_line_count(self) -> int:
        return len(self._log_buffer)

    def clear_log(self) -> None:
        self._log_buffer.clear()

    @property
    def pid(self) -> Optional[int]:
        return self.pm.pid

    # ── Engine / params setters (UI feeds these) ──────────────────────────────

    def set_engine(self, engine: str) -> None:
        self.engine = engine

    def set_params(self, params: dict) -> None:
        self._params = dict(params)

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = text or ""

    def set_gguf_path(self, path: str) -> None:
        self._gguf_path = path or ""

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self.pm.is_running

    @property
    def is_ollama(self) -> bool:
        return self.engine == ENGINE_OLLAMA

    @property
    def has_port(self) -> bool:
        """True if this engine exposes a chat-able HTTP port."""
        return self.engine in (ENGINE_LLAMA_SERVER, ENGINE_LLAMA_SWAP) or self.is_ollama

    # ── Config construction ──────────────────────────────────────────────────

    def build_config(self) -> ModelConfig:
        s = self.settings
        bin_dir = s.get_str("llama_bin_dir")
        mode = "cli" if self.engine == ENGINE_LLAMA_CLI else "server"
        cfg = ModelConfig.from_params(
            self._params,
            gguf_path=self._gguf_path,
            system_prompt=self._system_prompt,
            llama_mode=mode,
            llama_server_path=paths.resolve_llama_server(s.get_str("llama_server_path"), bin_dir),
            llama_cli_path=paths.resolve_llama_cli(s.get_str("llama_cli_path"), bin_dir),
            llama_swap_path=paths.resolve_llama_swap(s.get_str("llama_swap_path"), bin_dir),
            host=s.get_str("llama_host", "127.0.0.1"),
            port=s.get_int("llama_port", 8080),
            swap_listen=s.get_str("swap_listen", "127.0.0.1:8090"),
            n_gpu_layers=s.get_int("n_gpu_layers", 0),
            threads=s.get_int("threads", -1),
            flash_attn=s.get_bool("flash_attn", False),
            cont_batching=s.get_bool("cont_batching", True),
        )
        return cfg

    # ── Chat target resolution (EXPLICIT) ──────────────────────────────────────

    def chat_target(self) -> tuple[Backend, str, int, str]:
        """
        Return (backend, host, port, model) for the chat engine.
        For llama-swap, host/port are parsed from swap_listen.
        """
        s = self.settings
        if self.is_ollama:
            return (Backend.OLLAMA,
                    s.get_str("ollama_host", "127.0.0.1"),
                    s.get_int("ollama_port", 11434),
                    "")  # model set by UI for Ollama
        if self.engine == ENGINE_LLAMA_SWAP:
            listen = s.get_str("swap_listen", "127.0.0.1:8090")
            host, _, port = listen.rpartition(":")
            return (Backend.LLAMACPP, host or "127.0.0.1", int(port or 8080), "")
        # server (and cli has no port, but default here)
        return (Backend.LLAMACPP,
                s.get_str("llama_host", "127.0.0.1"),
                s.get_int("llama_port", 8080), "")

    def chat_reachable(self) -> bool:
        """True if the chat HTTP endpoint is currently accepting connections."""
        if not self.has_port:
            return False
        _backend, host, port, _model = self.chat_target()
        return port_open(host, port, timeout=0.5)

    # ── Preflight ──────────────────────────────────────────────────────────────

    def preflight(self, cfg: ModelConfig) -> memory.PreflightResult:
        return memory.preflight_check(
            cfg.gguf_path, cfg.num_ctx,
            headroom_gb=self.settings.get_float("ram_headroom_gb", 1.5),
            headroom_pct=self.settings.get_float("ram_headroom_pct", 0.12),
        )

    # ── Start / Stop ────────────────────────────────────────────────────────────

    def start(self, cfg: Optional[ModelConfig] = None, skip_preflight: bool = False
              ) -> tuple[bool, str]:
        """
        Start the backend process. Returns (started, message).

        - Ollama: nothing to start; returns (True, info).
        - llama.cpp/swap: validates binary + model, runs preflight, then spawns.
        - If already running, stops first (blocking) to avoid duplicates.
        """
        cfg = cfg or self.build_config()

        if self.is_ollama:
            return True, "Ollama is external — no process to start."

        # Validate model
        if not cfg.gguf_path:
            return False, "No GGUF model selected."
        if not paths.binary_exists(cfg.gguf_path):
            return False, f"Model file not found: {cfg.gguf_path}"

        # Validate binary
        if self.engine == ENGINE_LLAMA_SWAP:
            binary = cfg.llama_swap_path
            label = "llama-swap"
        elif cfg.llama_mode == "cli":
            binary = cfg.llama_cli_path
            label = "llama-cli"
        else:
            binary = cfg.llama_server_path
            label = "llama-server"
        if not binary:
            return False, f"{label} binary not found. Set its path in Settings."

        # Preflight (server/swap only; cli streams to stdout, less critical)
        if not skip_preflight and cfg.llama_mode != "cli":
            pf = self.preflight(cfg)
            if not pf.fits:
                return False, pf.message

        # Ensure no duplicate instance is running (blocking stop)
        if self.pm.is_running:
            self.pm.stop()

        # Build argv + start
        if self.engine == ENGINE_LLAMA_SWAP:
            import json, tempfile
            from pathlib import Path
            swap_cfg = lc.build_swap_config(cfg)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                              prefix="mf_swap_", delete=False,
                                              dir=str(Path.home()))
            json.dump(swap_cfg, tmp, indent=2)
            tmp.close()
            argv = [cfg.llama_swap_path, "--config", tmp.name]
            host, _, port = cfg.swap_listen.rpartition(":")
            self.pm.start(argv, mode="swap", host=host or "127.0.0.1", port=int(port or 8090))
        else:
            mode = cfg.llama_mode
            argv = lc.build_argv(cfg, mode=mode)
            self.pm.start(argv, mode=mode, host=cfg.host, port=cfg.port)

        return True, f"Starting {label}…"

    def stop(self) -> None:
        self.pm.stop_async()

    def stop_blocking(self) -> None:
        self.pm.stop()

    # ── Maintenance ──────────────────────────────────────────────────────────

    def kill_orphans(self) -> int:
        """Kill stray llama-server/cli/swap processes from prior crashes."""
        total = 0
        for name in ("llama-server", "llama-cli", "llama-swap"):
            total += kill_orphans(name)
        return total

    def command_preview(self) -> str:
        """Human-readable command line for the current config."""
        cfg = self.build_config()
        if self.is_ollama:
            return f"(Ollama @ {self.settings.get_str('ollama_host')}:{self.settings.get_int('ollama_port')})"
        if self.engine == ENGINE_LLAMA_SWAP:
            return lc.swap_yaml(cfg)
        return lc.build_command_str(cfg, mode=cfg.llama_mode)

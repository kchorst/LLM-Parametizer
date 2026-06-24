"""
llama.cpp adapter (PRIMARY backend).

Translates a ModelConfig into:
  - llama-server / llama-cli argv (build_argv)
  - llama-swap config dict / YAML (build_swap_config / swap_yaml)  [YAML is swap-only]
  - a portable INI profile (to_ini / from_ini), embedding the exact command line
  - an Ollama Modelfile (to_modelfile)  [secondary convenience]
"""

from __future__ import annotations

import configparser
import io
import shlex
from datetime import datetime
from pathlib import Path

from ..core.models import ModelConfig, PARAM_KEYS, default_params


# ── Sampling flag mapping ───────────────────────────────────────────────────────

_PARAM_TO_FLAG = {
    "temperature":    "--temp",
    "top_p":          "--top-p",
    "top_k":          "--top-k",
    "repeat_penalty": "--repeat-penalty",
    "num_ctx":        "--ctx-size",
    "num_predict":    "--n-predict",
    "seed":           "--seed",
}


def _sampling_args(cfg: ModelConfig) -> list[str]:
    args: list[str] = []
    params = cfg.params()
    for key, flag in _PARAM_TO_FLAG.items():
        val = params.get(key)
        if val is None:
            continue
        if key == "num_ctx":
            continue  # already added explicitly in build_argv
        if key == "num_predict" and val == -1:
            continue  # server default; omit
        if key == "seed" and val == 0:
            continue
        args += [flag, str(val)]
    if cfg.mirostat in (1, 2):
        args += ["--mirostat", str(cfg.mirostat), "--mirostat-tau", str(cfg.mirostat_tau)]
    return args


# ── argv builders ───────────────────────────────────────────────────────────────

def build_argv(cfg: ModelConfig, mode: str | None = None) -> list[str]:
    """Return argv for llama-server or llama-cli."""
    effective = mode or cfg.llama_mode

    if effective == "server":
        binary = cfg.llama_server_path or "llama-server"
        argv = [binary, "--model", cfg.gguf_path,
                "--host", cfg.host, "--port", str(cfg.port),
                "--ctx-size", str(cfg.num_ctx)]
        if cfg.n_gpu_layers != 0:
            argv += ["--n-gpu-layers", str(cfg.n_gpu_layers)]
        if cfg.threads > 0:
            argv += ["--threads", str(cfg.threads)]
        if cfg.flash_attn:
            argv += ["--flash-attn"]
        if cfg.cont_batching:
            argv += ["--cont-batching"]
        argv += _sampling_args(cfg)
    else:
        binary = cfg.llama_cli_path or "llama-cli"
        argv = [binary, "--model", cfg.gguf_path, "--ctx-size", str(cfg.num_ctx)]
        if cfg.n_gpu_layers != 0:
            argv += ["--n-gpu-layers", str(cfg.n_gpu_layers)]
        if cfg.threads > 0:
            argv += ["--threads", str(cfg.threads)]
        argv += _sampling_args(cfg)
        if cfg.system_prompt:
            argv += ["-p", cfg.system_prompt]

    argv += list(cfg.extra_llama_args)
    return argv


def build_command_str(cfg: ModelConfig, mode: str | None = None) -> str:
    """Shell-quoted command for display/copy."""
    return shlex.join(build_argv(cfg, mode))


# ── llama-swap ───────────────────────────────────────────────────────────────────

def build_swap_config(cfg: ModelConfig) -> dict:
    models: list[dict] = list(cfg.swap_models)
    if not models and cfg.gguf_path:
        server_argv = build_argv(cfg, mode="server")
        cmd = " ".join(server_argv)
        models.append({"name": Path(cfg.gguf_path).stem, "cmd": cmd})
    return {"listen": cfg.swap_listen, "models": models}


def swap_yaml(cfg: ModelConfig) -> str:
    try:
        import yaml  # type: ignore
        return yaml.dump(build_swap_config(cfg), default_flow_style=False, allow_unicode=True)
    except ImportError:
        import json
        return json.dumps(build_swap_config(cfg), indent=2)


# ── INI profile (portable save/load, embeds command line) ───────────────────────

def _esc(text: str) -> str:
    """Escape newlines so multi-line prompts survive a single INI value."""
    return (text or "").replace("\\", "\\\\").replace("\n", "\\n")


def _unesc(text: str) -> str:
    """Reverse _esc()."""
    out = []
    i = 0
    s = text or ""
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def to_ini(cfg: ModelConfig) -> str:
    """
    Serialise the config as a portable INI profile.

    Includes a header comment with the profile name, timestamp, and the exact
    command line, plus a [command] section for machine-stable copy/paste.
    The config is rebuilt from the structured sections on import; [command] is
    informational only.
    """
    cp = configparser.ConfigParser(interpolation=None)

    cp["model"] = {
        "gguf_path": cfg.gguf_path,
        "config_name": cfg.config_name,
        "tag": cfg.tag,
        "ollama_base_model": cfg.ollama_base_model,
    }
    cp["prompt"] = {
        "system_prompt": _esc(cfg.system_prompt),
        "template": _esc(cfg.template),
    }
    cp["parameters"] = {k: str(v) for k, v in cfg.params().items()}
    cp["server"] = {
        "mode": cfg.llama_mode,
        "host": cfg.host,
        "port": str(cfg.port),
        "n_gpu_layers": str(cfg.n_gpu_layers),
        "threads": str(cfg.threads),
        "flash_attn": str(cfg.flash_attn).lower(),
        "cont_batching": str(cfg.cont_batching).lower(),
        "extra_args": shlex.join(cfg.extra_llama_args) if cfg.extra_llama_args else "",
    }

    server_cmd = build_command_str(cfg, mode="server")
    cmd_section = {"server": server_cmd}
    if cfg.llama_mode == "cli":
        cmd_section["cli"] = build_command_str(cfg, mode="cli")
    cp["command"] = cmd_section

    buf = io.StringIO()
    name = cfg.config_name or (Path(cfg.gguf_path).stem if cfg.gguf_path else "profile")
    buf.write(f"# LLM Parametizer profile: {name}\n")
    buf.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    buf.write(f"# Command ({cfg.llama_mode}):\n")
    buf.write(f"#   {server_cmd}\n\n")
    cp.write(buf)
    return buf.getvalue()


def from_ini(text: str) -> ModelConfig:
    """Parse an INI profile into a ModelConfig. Tolerant of missing/garbage input."""
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(text or "")
    except configparser.Error:
        cp = configparser.ConfigParser(interpolation=None)  # fall back to all-defaults

    def section(name: str) -> dict:
        return dict(cp[name]) if cp.has_section(name) else {}

    model = section("model")
    prompt = section("prompt")
    raw_params = section("parameters")
    server = section("server")

    params = dict(default_params())
    params.update({k: v for k, v in raw_params.items() if k in PARAM_KEYS})

    def _int(d: dict, key: str, default: int) -> int:
        try:
            return int(d.get(key, default))
        except (TypeError, ValueError):
            return default

    def _bool(d: dict, key: str, default: bool) -> bool:
        val = d.get(key)
        if val is None:
            return default
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    extra = server.get("extra_args", "") or ""
    try:
        extra_args = shlex.split(extra)
    except ValueError:
        extra_args = []

    cfg = ModelConfig.from_params(
        params,
        gguf_path=model.get("gguf_path", "") or "",
        config_name=model.get("config_name", "") or "",
        tag=model.get("tag", "") or "",
        ollama_base_model=model.get("ollama_base_model", "") or "",
        system_prompt=_unesc(prompt.get("system_prompt", "")),
        template=_unesc(prompt.get("template", "")),
        llama_mode=server.get("mode", "server") or "server",
        host=server.get("host", "127.0.0.1") or "127.0.0.1",
        port=_int(server, "port", 8080),
        n_gpu_layers=_int(server, "n_gpu_layers", 0),
        threads=_int(server, "threads", -1),
        flash_attn=_bool(server, "flash_attn", False),
        cont_batching=_bool(server, "cont_batching", True),
        extra_llama_args=extra_args,
    )
    return cfg


# ── Ollama Modelfile (secondary convenience) ─────────────────────────────────────

_PARAM_TO_OLLAMA = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k",
    "repeat_penalty": "repeat_penalty", "num_ctx": "num_ctx",
    "num_predict": "num_predict", "mirostat": "mirostat",
    "mirostat_tau": "mirostat_tau", "seed": "seed",
}
_OLLAMA_DEFAULTS = default_params()


def to_modelfile(cfg: ModelConfig) -> str:
    """Generate an Ollama Modelfile string from the config."""
    base = cfg.ollama_base_model or (Path(cfg.gguf_path).stem if cfg.gguf_path else "unknown")
    lines = [f"FROM {base}"]

    if cfg.system_prompt:
        esc = cfg.system_prompt.replace('"""', '\\"\\"\\"')
        lines += ["", f'SYSTEM """{esc}"""']
    if cfg.template:
        esc = cfg.template.replace('"""', '\\"\\"\\"')
        lines += ["", f'TEMPLATE """{esc}"""']

    params = cfg.params()
    pl = []
    for key, directive in _PARAM_TO_OLLAMA.items():
        val = params.get(key)
        if val is None or val == _OLLAMA_DEFAULTS.get(key):
            continue
        pl.append(f"PARAMETER {directive} {val}")
    if pl:
        lines.append("")
        lines.extend(pl)

    return "\n".join(lines) + "\n"

"""
Chat streaming engine.

Provides a single stream_chat() generator that yields content tokens from either
backend, ending with a final ChatStats object. Backend selection is EXPLICIT
(no heuristic probing) — the caller passes a Backend enum, fixing the legacy
chat-routing ambiguity.

  - llama.cpp: OpenAI-compatible /v1/chat/completions (PRIMARY)
  - Ollama:    /api/chat                              (SECONDARY)

Yields:
    str            for each content chunk
    ChatStats      as the final item (sentinel; check isinstance)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import requests


class Backend(str, Enum):
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"


@dataclass
class ChatStats:
    elapsed: float = 0.0
    tokens: int = 0
    tokens_per_second: float | None = None
    ttft: float | None = None   # time to first token (seconds)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Parameter mapping for llama-server (OpenAI fields) ────────────────────────────

_OPENAI_KEYMAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "repeat_penalty": "repeat_penalty",
    "num_predict": "max_tokens",
    "seed": "seed",
}


def _openai_options(options: dict | None) -> dict:
    payload: dict = {}
    if not options:
        return payload
    for src, dst in _OPENAI_KEYMAP.items():
        val = options.get(src)
        if val is None:
            continue
        if src == "num_predict" and val in (-1, -2):
            continue
        payload[dst] = val
    return payload


# ── Public entry point ───────────────────────────────────────────────────────────

def stream_chat(
    backend: Backend,
    messages: list[dict],
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    model: str = "",
    options: dict | None = None,
    timeout: float = 120.0,
) -> Iterator[object]:
    """
    Stream a chat completion. Yields str chunks then a final ChatStats.

    backend == LLAMACPP → host/port point at llama-server
    backend == OLLAMA   → host/port point at Ollama; model is required
    """
    if backend == Backend.LLAMACPP:
        yield from _stream_llamacpp(messages, host, port, options, timeout)
    else:
        yield from _stream_ollama(messages, host, port, model, options, timeout)


# ── llama.cpp (OpenAI-compatible) ─────────────────────────────────────────────────

def _stream_llamacpp(messages, host, port, options, timeout) -> Iterator[object]:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload: dict = {"messages": messages, "stream": True}
    payload.update(_openai_options(options))

    start = time.time()
    tokens = 0
    first_at: float | None = None
    try:
        with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                detail = _extract_error(r)
                yield ChatStats(elapsed=_elapsed(start), error=f"HTTP {r.status_code}: {detail}")
                return
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content", "")
                    if content:
                        if first_at is None:
                            first_at = time.time()
                        tokens += 1
                        yield content
        yield ChatStats(elapsed=_elapsed(start), tokens=tokens,
                        tokens_per_second=_tps(tokens, start),
                        ttft=_ttft(start, first_at))
    except requests.exceptions.ConnectionError:
        yield ChatStats(elapsed=_elapsed(start), error="Connection failed — is llama-server running?")
    except Exception as e:
        yield ChatStats(elapsed=_elapsed(start), error=str(e))


# ── Ollama ────────────────────────────────────────────────────────────────────────

def _stream_ollama(messages, host, port, model, options, timeout) -> Iterator[object]:
    if not model:
        yield ChatStats(error="No Ollama model specified.")
        return
    url = f"http://{host}:{port}/api/chat"
    payload: dict = {"model": model, "messages": messages, "stream": True}
    if options:
        payload["options"] = options

    start = time.time()
    tokens = 0
    first_at: float | None = None
    try:
        with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                detail = _extract_error(r)
                yield ChatStats(elapsed=_elapsed(start), error=f"HTTP {r.status_code}: {detail}")
                return
            for raw in r.iter_lines():
                if not raw:
                    continue
                try:
                    data = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                msg = data.get("message") or {}
                content = msg.get("content", "")
                if content:
                    if first_at is None:
                        first_at = time.time()
                    tokens += 1
                    yield content
        yield ChatStats(elapsed=_elapsed(start), tokens=tokens,
                        tokens_per_second=_tps(tokens, start),
                        ttft=_ttft(start, first_at))
    except requests.exceptions.ConnectionError:
        yield ChatStats(elapsed=_elapsed(start), error="Connection failed — is Ollama running?")
    except Exception as e:
        yield ChatStats(elapsed=_elapsed(start), error=str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _elapsed(start: float) -> float:
    return round(time.time() - start, 3)


def _tps(tokens: int, start: float) -> float | None:
    el = time.time() - start
    return round(tokens / el, 1) if tokens and el > 0 else None


def _ttft(start: float, first_at: float | None) -> float | None:
    return round(first_at - start, 3) if first_at is not None else None


def _extract_error(r) -> str:
    try:
        body = r.text
        try:
            j = json.loads(body)
            return j.get("error", {}).get("message", body[:200]) if isinstance(j.get("error"), dict) else str(j.get("error", body[:200]))
        except Exception:
            return body[:200]
    except Exception:
        return r.reason or "unknown error"

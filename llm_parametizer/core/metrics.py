"""
Performance metrics: session accumulation + config snapshots + hardware probe.

The consultant persona needs hard evidence (before/after) to justify a tuned
configuration to a client. This module provides:

  - SessionMetrics : running accumulator of per-message ChatStats (TTFT, TPS,
                     tokens, elapsed) with averages and min/max.
  - Snapshot       : a frozen capture of a config + its measured performance and
                     (optionally) quality metrics, used for before/after compare.
  - hardware_summary() : best-effort CPU/RAM/GPU description for the report header.

Everything here is UI-free and import-light so it can be unit-tested in isolation.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime

from . import memory


# ── Session accumulation ──────────────────────────────────────────────────────

@dataclass
class SessionMetrics:
    """Accumulates per-message performance samples over a chat session."""
    count: int = 0
    total_tokens: int = 0
    _ttfts: list[float] = field(default_factory=list)
    _tpss: list[float] = field(default_factory=list)
    _elapseds: list[float] = field(default_factory=list)

    def add(self, stats) -> None:
        """Record one ChatStats. Ignores errored or empty results."""
        if stats is None or getattr(stats, "error", None):
            return
        if not getattr(stats, "tokens", 0):
            return
        self.count += 1
        self.total_tokens += int(stats.tokens)
        if getattr(stats, "ttft", None) is not None:
            self._ttfts.append(float(stats.ttft))
        if getattr(stats, "tokens_per_second", None):
            self._tpss.append(float(stats.tokens_per_second))
        if getattr(stats, "elapsed", None):
            self._elapseds.append(float(stats.elapsed))

    def reset(self) -> None:
        self.count = 0
        self.total_tokens = 0
        self._ttfts.clear()
        self._tpss.clear()
        self._elapseds.clear()

    # ── Aggregates (None when no samples) ─────────────────────────────────────

    @staticmethod
    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 2) if xs else None

    @property
    def avg_ttft(self) -> float | None:
        return self._avg(self._ttfts)

    @property
    def avg_tps(self) -> float | None:
        return self._avg(self._tpss)

    @property
    def avg_elapsed(self) -> float | None:
        return self._avg(self._elapseds)

    @property
    def min_ttft(self) -> float | None:
        return round(min(self._ttfts), 2) if self._ttfts else None

    @property
    def max_ttft(self) -> float | None:
        return round(max(self._ttfts), 2) if self._ttfts else None

    @property
    def min_tps(self) -> float | None:
        return round(min(self._tpss), 1) if self._tpss else None

    @property
    def max_tps(self) -> float | None:
        return round(max(self._tpss), 1) if self._tpss else None

    def summary(self) -> dict:
        """Flat dict of the aggregates (for snapshots / reports)."""
        return {
            "messages": self.count,
            "total_tokens": self.total_tokens,
            "avg_ttft": self.avg_ttft,
            "min_ttft": self.min_ttft,
            "max_ttft": self.max_ttft,
            "avg_tps": self.avg_tps,
            "min_tps": self.min_tps,
            "max_tps": self.max_tps,
            "avg_elapsed": self.avg_elapsed,
        }


# ── Snapshots (before/after) ──────────────────────────────────────────────────

@dataclass
class Snapshot:
    """A frozen capture of a configuration and its measured metrics.

    `perf` holds performance aggregates (avg_tps, avg_ttft, total_tokens, …).
    `quality` holds optional quality metrics from a tune run (score, repetition,
    refusals). Either may be partial or empty.
    """
    label: str
    model_name: str = ""
    family: str = ""
    goal: str = ""
    params: dict = field(default_factory=dict)
    command: str = ""
    perf: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    def metric(self, key: str):
        """Look up a metric from perf first, then quality."""
        if key in self.perf:
            return self.perf[key]
        return self.quality.get(key)


# ── Hardware probe ─────────────────────────────────────────────────────────────

def hardware_summary() -> dict:
    """Best-effort CPU / RAM / GPU description for the report header."""
    total, avail = memory.system_memory()
    vram = memory.vram_info()
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "ram_total_gb": round(total / 1024 ** 3, 1) if total else None,
        "ram_available_gb": round(avail / 1024 ** 3, 1) if avail else None,
        "gpu": vram.name if vram.available else None,
        "vram_total_gb": round(vram.total_bytes / 1024 ** 3, 1) if vram.available and vram.total_bytes else None,
    }
    return info

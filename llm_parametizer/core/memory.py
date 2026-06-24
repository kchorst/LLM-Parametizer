"""
Memory estimation, system memory queries, and cleanup utilities.

Provides:
  - system_memory()      → (total, available) bytes
  - vram_info()          → optional GPU VRAM via nvidia-smi
  - estimate_model_ram() → estimated footprint for a ModelConfig
  - preflight_check()    → does the model fit in available RAM?
  - flush_memory()       → force GC and trim working set
"""

from __future__ import annotations

import gc
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── System memory ───────────────────────────────────────────────────────────────

def system_memory() -> tuple[int, int]:
    """
    Return (total_bytes, available_bytes).
    Falls back to (0, 0) if psutil is unavailable.
    """
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:
            pass
    return 0, 0


@dataclass
class VRAMInfo:
    available: bool
    total_bytes: int = 0
    used_bytes: int = 0
    name: str = ""

    @property
    def free_bytes(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)


def vram_info() -> VRAMInfo:
    """
    Query GPU VRAM via nvidia-smi. Returns VRAMInfo(available=False) if no NVIDIA
    GPU is detected or the tool is missing.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.used,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return VRAMInfo(available=False)
        # Use the first GPU line
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        total_mb = float(parts[0])
        used_mb = float(parts[1])
        name = parts[2] if len(parts) > 2 else ""
        return VRAMInfo(
            available=True,
            total_bytes=int(total_mb * 1024 ** 2),
            used_bytes=int(used_mb * 1024 ** 2),
            name=name,
        )
    except Exception:
        return VRAMInfo(available=False)


# ── GGUF metadata: layer count ───────────────────────────────────────────────────

def read_gguf_block_count(path: str | Path) -> int | None:
    """
    Read the *.block_count key from a GGUF header (number of transformer layers).
    Returns None if the file cannot be parsed or the key is absent.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            struct.unpack("<I", f.read(4))[0]   # version
            struct.unpack("<Q", f.read(8))[0]   # n_tensors
            n_kv = struct.unpack("<Q", f.read(8))[0]

            for _ in range(n_kv):
                key_len = struct.unpack("<Q", f.read(8))[0]
                if key_len > 1024:
                    return None
                key = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                val = _read_gguf_value(f, vtype)
                if val is None and vtype == 9:
                    return None  # hit an array we can't skip safely
                if "block_count" in key and isinstance(val, int):
                    return int(val)
    except Exception:
        return None
    return None


def _read_gguf_value(f, vtype: int):
    """Read a single GGUF metadata value. Returns the int/float value or None."""
    # 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool 8=str 9=array 10=u64 11=i64 12=f64
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 6:
        f.read(4); return None
    if vtype == 12:
        f.read(8); return None
    if vtype == 7:
        f.read(1); return None
    if vtype == 8:
        slen = struct.unpack("<Q", f.read(8))[0]
        if slen > 1 << 20:
            return None
        f.read(slen); return None
    if vtype in (0, 1):
        f.read(1); return None
    if vtype in (2, 3):
        f.read(2); return None
    if vtype == 9:
        return None  # array — caller treats as unparseable
    return None


# ── Estimation ───────────────────────────────────────────────────────────────────

@dataclass
class RAMEstimate:
    model_bytes: int
    kv_bytes: int
    total_bytes: int
    n_layers: int
    explanation: str

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1024 ** 3


def estimate_model_ram(gguf_path: str, num_ctx: int) -> RAMEstimate:
    """
    Estimate RAM required to load a model at a given context size.

    model_bytes   = GGUF file size on disk
    kv_bytes      = num_ctx * per_token, where per_token scales with layer count
    """
    model_bytes = 0
    p = Path(gguf_path) if gguf_path else None
    if p and p.is_file():
        try:
            model_bytes = p.stat().st_size
        except Exception:
            model_bytes = 0

    n_layers = read_gguf_block_count(gguf_path) or 32

    # KV-cache per token: ~256 bytes per layer (f16 k+v, typical head dims),
    # floored to avoid absurdly small estimates.
    kv_per_token = max(512, n_layers * 256)
    kv_bytes = max(0, int(num_ctx)) * kv_per_token

    total = model_bytes + kv_bytes

    parts = []
    if model_bytes:
        parts.append(f"model {model_bytes / 1024**3:.2f} GB")
    parts.append(f"KV ~{kv_bytes / 1024**2:.0f} MB (ctx={num_ctx}, layers={n_layers})")
    explanation = " + ".join(parts) + f" ≈ {total / 1024**3:.2f} GB"

    return RAMEstimate(model_bytes, kv_bytes, total, n_layers, explanation)


@dataclass
class PreflightResult:
    fits: bool
    estimate: RAMEstimate
    available_bytes: int
    headroom_bytes: int
    budget_bytes: int
    message: str

    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / 1024 ** 3

    @property
    def available_gb(self) -> float:
        return self.available_bytes / 1024 ** 3


def preflight_check(
    gguf_path: str,
    num_ctx: int,
    headroom_gb: float = 1.5,
    headroom_pct: float = 0.12,
) -> PreflightResult:
    """
    Check whether a model at the given context fits in available RAM.

    Budget = available - max(headroom_gb, headroom_pct * total).
    If system memory is unreadable, fits defaults to True (don't block).
    """
    total, available = system_memory()
    est = estimate_model_ram(gguf_path, num_ctx)

    if not available:
        return PreflightResult(
            fits=True, estimate=est, available_bytes=0,
            headroom_bytes=0, budget_bytes=0,
            message="System memory unreadable — skipping preflight.",
        )

    headroom = max(int(headroom_gb * 1024 ** 3), int((total or available) * headroom_pct))
    budget = max(0, available - headroom)
    fits = est.total_bytes <= budget

    def gb(n: float) -> str:
        return f"{n / 1024**3:.1f} GB"

    if fits:
        msg = f"OK — needs {gb(est.total_bytes)}, budget {gb(budget)}."
    else:
        msg = (f"Insufficient RAM — needs {gb(est.total_bytes)} ({est.explanation}); "
               f"budget {gb(budget)} (free {gb(available)} − headroom {gb(headroom)}).")

    return PreflightResult(fits, est, available, headroom, budget, msg)


def largest_fitting_ctx(
    gguf_path: str,
    desired_ctx: int,
    headroom_gb: float = 1.5,
    headroom_pct: float = 0.12,
    candidates: tuple[int, ...] = (32768, 16384, 12288, 8192, 6144, 4096, 3072, 2048),
) -> int | None:
    """
    Find the largest standard context size <= desired_ctx that fits in RAM.
    Returns None if even the smallest candidate does not fit.
    """
    for c in candidates:
        if c > desired_ctx:
            continue
        if preflight_check(gguf_path, c, headroom_gb, headroom_pct).fits:
            return c
    return None


# ── Cleanup ───────────────────────────────────────────────────────────────────

def flush_memory() -> int:
    """
    Force Python garbage collection and trim the process working set on Windows.
    Returns the number of objects collected by gc.
    """
    collected = gc.collect()
    try:
        import ctypes
        if hasattr(ctypes, "windll"):
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)  # type: ignore[attr-defined]
    except Exception:
        pass
    return collected

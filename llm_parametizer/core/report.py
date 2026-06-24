"""
Delivery report generation.

Builds a client-facing report from one or two Snapshots (before/after). Ported
and modernised from the legacy `build_delivery_report`, now including measured
performance (TTFT, TPS, tokens), optional quality metrics, a hardware header,
and a before/after comparison with deltas.

Two renderers:
  - build_markdown() : rich Markdown for the deliverable file
  - build_text()     : monospace plain-text (legacy look), good for clipboard

Both accept the same inputs so the UI can offer either format.
"""

from __future__ import annotations

from datetime import datetime

from .metrics import Snapshot
from . import families


# ── Metric display registry ────────────────────────────────────────────────────
# key → (label, unit, higher_is_better)
_METRICS = [
    ("avg_ttft", "Time to first token", "s", False),
    ("avg_tps", "Tokens / second", "tok/s", True),
    ("total_tokens", "Total tokens", "", True),
    ("avg_elapsed", "Avg response time", "s", False),
    ("score", "Quality score", "/100", True),
    ("repetition", "Repetition", "", False),
    ("refusals", "Refusals", "", False),
]


def _fmt(val, unit: str) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        val = f"{val:g}"
    return f"{val}{(' ' + unit) if unit else ''}".strip()


def _delta(before, after, higher_is_better: bool) -> str:
    """Human delta string with direction arrow, or '' when not comparable."""
    if before is None or after is None:
        return ""
    try:
        diff = after - before
    except TypeError:
        return ""
    if abs(diff) < 1e-9:
        return "no change"
    better = (diff > 0) == higher_is_better
    arrow = "▲" if diff > 0 else "▼"
    pct = ""
    if before:
        pct = f" ({diff / before * 100:+.0f}%)"
    tag = "better" if better else "worse"
    return f"{arrow} {diff:+g}{pct} — {tag}"


def family_notes(model_name: str, goal: str = "") -> list[str]:
    """Family-specific guidance for the report's notes section."""
    fam = families.detect_family(model_name)
    notes = list(families.constraints_for(model_name).get("notes", []))
    notes.insert(0, f"Model family detected: {fam}")
    if goal:
        notes.append(f"Optimization goal: {goal}.")
    return notes


# ── Markdown renderer ───────────────────────────────────────────────────────────

def build_markdown(
    snapshots: list[Snapshot],
    *,
    hardware: dict | None = None,
    session_notes: str = "",
    client: str = "",
) -> str:
    if not snapshots:
        return "# LLM Parametizer — Report\n\n_No configuration captured yet._\n"

    primary = snapshots[-1]
    lines: list[str] = []
    lines.append("# LLM Parametizer — Model Delivery Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if client:
        lines.append(f"- **Client:** {client}")
    lines.append(f"- **Model:** `{primary.model_name or 'unknown'}`")
    lines.append(f"- **Family:** {primary.family or families.detect_family(primary.model_name)}")
    if primary.goal:
        lines.append(f"- **Goal:** {primary.goal}")
    lines.append("")

    # Hardware
    if hardware:
        lines.append("## Test Environment")
        lines.append("")
        hw = hardware
        if hw.get("os"):
            lines.append(f"- **OS:** {hw['os']}")
        if hw.get("cpu"):
            lines.append(f"- **CPU:** {hw['cpu']}")
        if hw.get("ram_total_gb"):
            lines.append(f"- **RAM:** {hw['ram_total_gb']} GB")
        if hw.get("gpu"):
            vram = f" ({hw['vram_total_gb']} GB)" if hw.get("vram_total_gb") else ""
            lines.append(f"- **GPU:** {hw['gpu']}{vram}")
        lines.append("")

    # Metrics — before/after table when two snapshots, else single column.
    lines.append("## Performance & Quality")
    lines.append("")
    if len(snapshots) >= 2:
        before, after = snapshots[0], snapshots[-1]
        lines.append(f"| Metric | {before.label} | {after.label} | Change |")
        lines.append("|---|---|---|---|")
        for key, label, unit, hib in _METRICS:
            b, a = before.metric(key), after.metric(key)
            if b is None and a is None:
                continue
            lines.append(f"| {label} | {_fmt(b, unit)} | {_fmt(a, unit)} | {_delta(b, a, hib) or '—'} |")
    else:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key, label, unit, _hib in _METRICS:
            v = primary.metric(key)
            if v is None:
                continue
            lines.append(f"| {label} | {_fmt(v, unit)} |")
    lines.append("")

    # Parameters
    lines.append("## Recommended Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    for k, v in sorted(primary.params.items()):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    # Launch command
    if primary.command:
        lines.append("## Launch Command")
        lines.append("")
        lines.append("```bash")
        lines.append(primary.command)
        lines.append("```")
        lines.append("")

    # Notes
    notes = primary.notes or family_notes(primary.model_name, primary.goal)
    if notes:
        lines.append("## Tuning Notes")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    if session_notes and session_notes.strip():
        lines.append("## Session Notes")
        lines.append("")
        lines.append(session_notes.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── Plain-text renderer (legacy look) ────────────────────────────────────────────

def build_text(
    snapshots: list[Snapshot],
    *,
    hardware: dict | None = None,
    session_notes: str = "",
    client: str = "",
) -> str:
    if not snapshots:
        return "LLM PARAMETIZER — REPORT\n\nNo configuration captured yet.\n"

    primary = snapshots[-1]
    W = 60
    out: list[str] = []
    out.append("=" * W)
    out.append("  LLM PARAMETIZER — MODEL DELIVERY REPORT")
    out.append("=" * W)
    out.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if client:
        out.append(f"  Client    : {client}")
    out.append(f"  Model     : {primary.model_name or 'unknown'}")
    out.append(f"  Family    : {primary.family or families.detect_family(primary.model_name)}")
    if primary.goal:
        out.append(f"  Goal      : {primary.goal}")

    if hardware:
        out += ["", "─" * W, "  TEST ENVIRONMENT", "─" * W]
        if hardware.get("os"):
            out.append(f"  OS   : {hardware['os']}")
        if hardware.get("cpu"):
            out.append(f"  CPU  : {hardware['cpu']}")
        if hardware.get("ram_total_gb"):
            out.append(f"  RAM  : {hardware['ram_total_gb']} GB")
        if hardware.get("gpu"):
            vram = f" ({hardware['vram_total_gb']} GB)" if hardware.get("vram_total_gb") else ""
            out.append(f"  GPU  : {hardware['gpu']}{vram}")

    out += ["", "─" * W, "  PERFORMANCE & QUALITY", "─" * W]
    if len(snapshots) >= 2:
        before, after = snapshots[0], snapshots[-1]
        out.append(f"  {'Metric':<22}{before.label:>14}{after.label:>14}")
        for key, label, unit, hib in _METRICS:
            b, a = before.metric(key), after.metric(key)
            if b is None and a is None:
                continue
            out.append(f"  {label:<22}{_fmt(b, unit):>14}{_fmt(a, unit):>14}")
            d = _delta(b, a, hib)
            if d:
                out.append(f"  {'':<22}{('→ ' + d):>28}")
    else:
        for key, label, unit, _hib in _METRICS:
            v = primary.metric(key)
            if v is None:
                continue
            out.append(f"  {label:<24} {_fmt(v, unit)}")

    out += ["", "─" * W, "  RECOMMENDED PARAMETERS", "─" * W]
    for k, v in sorted(primary.params.items()):
        out.append(f"  {k:<24} {v}")

    if primary.command:
        out += ["", "─" * W, "  LAUNCH COMMAND", "─" * W, f"  {primary.command}"]

    notes = primary.notes or family_notes(primary.model_name, primary.goal)
    if notes:
        out += ["", "─" * W, "  TUNING NOTES", "─" * W]
        for n in notes:
            out.append(f"  • {n}")

    if session_notes and session_notes.strip():
        out += ["", "─" * W, "  SESSION NOTES", "─" * W, session_notes.strip()]

    out += ["", "=" * W, ""]
    return "\n".join(out)

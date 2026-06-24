"""
Model search + download (Hugging Face Hub, GGUF-focused).

Pure-ish IO layer over the public Hugging Face HTTP API — no auth required for
public repos and no extra dependencies (uses `requests`, already a dependency):

  - search_models()     find GGUF model repos by free-text query
  - list_gguf_files()   list the .gguf files (name + size) inside a repo
  - resolve_url()        build the direct download URL for a file
  - Downloader          threaded, cancellable streaming download with progress

The UI (ui/model_download_dialog.py) drives these; this module never touches Tk.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

HF_API = "https://huggingface.co/api"
HF_BASE = "https://huggingface.co"
_TIMEOUT = 20.0
_USER_AGENT = "ModelForge/0.1 (+https://localhost)"


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class RepoHit:
    repo_id: str
    downloads: int = 0
    likes: int = 0

    @property
    def author(self) -> str:
        return self.repo_id.split("/", 1)[0] if "/" in self.repo_id else ""


@dataclass
class GgufFile:
    repo_id: str
    path: str          # filename within the repo (may include subdir)
    size: int = 0      # bytes (0 if unknown)

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def size_h(self) -> str:
        return human_size(self.size)


def human_size(n: int) -> str:
    if not n:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}" if u != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TB"


# ── Search / listing ────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"User-Agent": _USER_AGENT, "Accept": "application/json"}


def search_models(query: str, limit: int = 25) -> list[RepoHit]:
    """Search Hugging Face for GGUF model repos matching `query`, by downloads."""
    params = {
        "search": query.strip(),
        "filter": "gguf",
        "limit": str(max(1, min(limit, 100))),
        "sort": "downloads",
        "direction": "-1",
        "full": "false",
    }
    r = requests.get(f"{HF_API}/models", params=params, headers=_headers(), timeout=_TIMEOUT)
    r.raise_for_status()
    out: list[RepoHit] = []
    for item in r.json():
        rid = item.get("id") or item.get("modelId") or ""
        if not rid:
            continue
        out.append(RepoHit(repo_id=rid,
                           downloads=int(item.get("downloads", 0) or 0),
                           likes=int(item.get("likes", 0) or 0)))
    return out


def list_gguf_files(repo_id: str) -> list[GgufFile]:
    """List .gguf files (with sizes) in a repo's main branch."""
    url = f"{HF_API}/models/{repo_id}/tree/main"
    r = requests.get(url, params={"recursive": "true"}, headers=_headers(), timeout=_TIMEOUT)
    r.raise_for_status()
    files: list[GgufFile] = []
    for ent in r.json():
        if ent.get("type") != "file":
            continue
        path = ent.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        # LFS files carry their real size under "lfs"; plain files under "size".
        size = 0
        lfs = ent.get("lfs") or {}
        if isinstance(lfs, dict) and lfs.get("size"):
            size = int(lfs["size"])
        elif ent.get("size"):
            size = int(ent["size"])
        files.append(GgufFile(repo_id=repo_id, path=path, size=size))
    files.sort(key=lambda f: f.filename.lower())
    return files


def resolve_url(repo_id: str, path: str) -> str:
    return f"{HF_BASE}/{repo_id}/resolve/main/{path}"


# ── Download ──────────────────────────────────────────────────────────────────

class Downloader:
    """Threaded, cancellable streaming download.

    Callbacks are invoked from the worker thread; the UI must marshal them onto
    the main thread (e.g. widget.after(0, ...)).
    """

    def __init__(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,   # (downloaded, total)
        on_done: Optional[Callable[[str], None]] = None,            # (final_path)
        on_error: Optional[Callable[[str], None]] = None,           # (message)
    ):
        self.on_progress = on_progress or (lambda d, t: None)
        self.on_done = on_done or (lambda p: None)
        self.on_error = on_error or (lambda m: None)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()

    def start(self, repo_id: str, path: str, dest_dir: str) -> bool:
        if self.is_running:
            return False
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run, args=(repo_id, path, dest_dir), daemon=True)
        self._thread.start()
        return True

    def _run(self, repo_id: str, path: str, dest_dir: str):
        url = resolve_url(repo_id, path)
        dest = Path(dest_dir)
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.on_error(f"Cannot create folder: {e}")
            return

        final = dest / Path(path).name
        tmp = final.with_suffix(final.suffix + ".part")
        try:
            with requests.get(url, headers={"User-Agent": _USER_AGENT}, stream=True,
                              timeout=_TIMEOUT, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0) or 0)
                done = 0
                self.on_progress(0, total)
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                        if self._cancel.is_set():
                            fh.close()
                            try:
                                tmp.unlink(missing_ok=True)
                            except Exception:
                                pass
                            self.on_error("Cancelled.")
                            return
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        self.on_progress(done, total)
            tmp.replace(final)
            self.on_done(str(final))
        except Exception as e:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            self.on_error(str(e))

"""
ProcessManager — robust single-process lifecycle for backend servers.

Design goals (fixes the legacy zombie/race bugs):
  - Exactly ONE managed process at a time.
  - Blocking stop() that truly waits, plus stop_async() helper.
  - _proc is ALWAYS cleared when the process ends (success OR error).
  - Port-ready polling for server mode before declaring "ready".
  - Singleton enforcement: kill_orphans() removes stray binaries by name.

Callbacks (invoked on daemon threads, never assumed to be on a UI thread):
  on_log(line)      — a new stdout/stderr line
  on_status(status) — "starting" | "ready" | "stopping" | "stopped" | "error"
  on_ready()        — server port is responding (server/swap mode only)
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
from typing import Callable, Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _no_window_kwargs() -> dict:
    """Windows: prevent a console window and allow CTRL_BREAK signalling."""
    if os.name != "nt":
        return {}
    kw: dict = {}
    try:
        kw["creationflags"] = (
            subprocess.CREATE_NO_WINDOW |          # type: ignore[attr-defined]
            subprocess.CREATE_NEW_PROCESS_GROUP    # type: ignore[attr-defined]
        )
    except AttributeError:
        pass
    try:
        si = subprocess.STARTUPINFO()              # type: ignore[attr-defined]
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
        si.wShowWindow = 0
        kw["startupinfo"] = si
    except AttributeError:
        pass
    return kw


def kill_tree(pid: int, timeout: float = 3.0) -> None:
    """Kill a process and all its children."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=timeout)
        except Exception:
            pass
    else:
        try:
            import psutil  # type: ignore
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def kill_orphans(binary_name: str) -> int:
    """
    Kill any stray processes matching binary_name (e.g. 'llama-server').
    Returns the number of kill attempts made. Best-effort; never raises.
    """
    count = 0
    if os.name == "nt":
        image = binary_name if binary_name.lower().endswith(".exe") else binary_name + ".exe"
        try:
            r = subprocess.run(["taskkill", "/F", "/IM", image],
                               capture_output=True, text=True, timeout=5)
            if "SUCCESS" in (r.stdout or ""):
                count = r.stdout.count("SUCCESS")
        except Exception:
            pass
    else:
        try:
            import psutil  # type: ignore
            for proc in psutil.process_iter(["name"]):
                try:
                    if binary_name in (proc.info.get("name") or ""):
                        proc.kill()
                        count += 1
                except Exception:
                    pass
        except Exception:
            pass
    return count


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── ProcessManager ────────────────────────────────────────────────────────────

class ProcessManager:
    STOP_GRACE_SECONDS = 5.0

    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log_thread: Optional[threading.Thread] = None
        self._ready_thread: Optional[threading.Thread] = None
        self._stopping = False  # True while an intentional stop is in progress

        self.on_log = on_log or (lambda _l: None)
        self.on_status = on_status or (lambda _s: None)
        self.on_ready = on_ready or (lambda: None)

        self._host = "127.0.0.1"
        self._port = 8080
        self._mode = "server"

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> Optional[int]:
        with self._lock:
            return self._proc.pid if self._proc else None

    # ── Start ──────────────────────────────────────────────────────────────────

    def start(
        self,
        argv: list[str],
        mode: str = "server",
        host: str = "127.0.0.1",
        port: int = 8080,
        ready_timeout: float = 60.0,
    ) -> None:
        """
        Spawn the process. Raises RuntimeError if one is already running.

        mode "server"/"swap" → poll port then fire on_ready.
        mode "cli"           → fire on_ready immediately (one-shot job).
        """
        if self.is_running:
            raise RuntimeError("A process is already running; stop it first.")

        self._host, self._port, self._mode = host, port, mode
        self.on_status("starting")
        self._emit_log(f"[pm] starting: {' '.join(argv)}")

        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **_no_window_kwargs(),
        )
        with self._lock:
            self._proc = proc

        self._log_thread = threading.Thread(target=self._stream_logs, args=(proc,), daemon=True)
        self._log_thread.start()

        if mode == "cli":
            self.on_status("ready")
            self.on_ready()
        else:
            self._ready_thread = threading.Thread(
                target=self._wait_for_port, args=(host, port, ready_timeout, proc), daemon=True
            )
            self._ready_thread.start()

    # ── Stop ───────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Blocking: graceful signal → grace period → force kill tree."""
        with self._lock:
            proc = self._proc
        if proc is None:
            return

        pid = proc.pid
        self._stopping = True
        self._emit_log(f"[pm] stopping pid {pid}")
        self.on_status("stopping")

        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                proc.terminate()
        except Exception:
            pass

        deadline = time.monotonic() + self.STOP_GRACE_SECONDS
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        if proc.poll() is None:
            self._emit_log("[pm] grace elapsed — force killing tree")
            kill_tree(pid)
            # Wait briefly for kill_tree to take effect
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
            # Final fallback if still alive
            if proc.poll() is None:
                self._emit_log("[pm] kill_tree failed — trying direct kill")
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass

        with self._lock:
            if self._proc is proc:
                self._proc = None

        self._stopping = False
        self.on_status("stopped")
        self._emit_log("[pm] stopped")

    def stop_async(self) -> None:
        """Non-blocking stop on a daemon thread."""
        threading.Thread(target=self.stop, daemon=True).start()

    # ── Internals ────────────────────────────────────────────────────────────

    def _emit_log(self, line: str) -> None:
        try:
            self.on_log(line)
        except Exception:
            pass

    def _stream_logs(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._emit_log(line.rstrip("\n"))
        except Exception:
            pass
        # Reap the process to obtain the true exit code (poll() can be None
        # immediately after stdout closes).
        try:
            ret = proc.wait(timeout=5)
        except Exception:
            ret = proc.poll()
        # Process ended — ALWAYS clear _proc if it's still this one.
        with self._lock:
            intentional = self._stopping or self._proc is None
            if self._proc is proc:
                self._proc = None
        if ret is not None and ret != 0 and not intentional:
            self._emit_log(f"[pm] exited with code {ret}")
            self.on_status("error")
        else:
            self.on_status("stopped")

    def _wait_for_port(self, host: str, port: int, timeout: float, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._emit_log("[pm] process exited before port was ready")
                self.on_status("error")
                return
            if port_open(host, port):
                self._emit_log(f"[pm] port {host}:{port} ready")
                self.on_status("ready")
                self.on_ready()
                return
            time.sleep(0.5)
        self._emit_log(f"[pm] timed out waiting for {host}:{port} after {timeout:.0f}s")
        self.on_status("error")

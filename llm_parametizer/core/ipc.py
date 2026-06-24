"""
IPC bridge — lightweight local HTTP endpoint for cross-app communication.

Accepts JSON messages of the form:

    { "timestamp": <number>, "data": { "action": "<allowed_action>", ... } }

from peer applications (e.g. LLM Shield browser extension relay,
LLM Tester) on a localhost-only port.

Design:
  - Runs in a daemon thread alongside the GUI.
  - Allowed actions are explicitly whitelisted (proxy allow-list).
  - Only binds to 127.0.0.1 — never exposed to the network.
  - Handlers are registered by the app at startup; unrecognised actions
    are rejected with 403.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Optional

# Default port for the IPC bridge (configurable via settings)
DEFAULT_IPC_PORT = 9820

# ── Allowed actions (proxy allow-list) ────────────────────────────────────────

ALLOWED_ACTIONS: set[str] = {
    "ping",                   # health check
    "get_config",             # export current parameter config
    "set_config",             # import parameter config from Shield/Tester
    "get_status",             # backend status (running, model, engine)
    "get_model_info",         # active model metadata
    "export_profile",         # export a saved profile by name
    "shield_heartbeat",       # Shield extension heartbeat
    "report_metrics",         # inbound safe run metrics from LLM Tester
    "get_metrics",            # read the last shared run metrics
}

# ── Handler registry ──────────────────────────────────────────────────────────

ActionHandler = Callable[[dict], dict]
_handlers: dict[str, ActionHandler] = {}


def register_handler(action: str, handler: ActionHandler) -> None:
    """Register a handler for an allowed IPC action."""
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Action '{action}' is not in the allowed-list. "
                         f"Add it to ALLOWED_ACTIONS first.")
    _handlers[action] = handler


# Optional provider of safe status fields for the GET /ipc/status probe. The app
# registers this so peer extensions (Sentinel+, LLM Tester) can detect the app
# and read safe status without a POST. Must return only allow-listed safe fields.
_status_provider: Optional[Callable[[], dict]] = None


def register_status_provider(provider: Callable[[], dict]) -> None:
    """Register a callable returning safe status fields for GET /ipc/status."""
    global _status_provider
    _status_provider = provider


def _dispatch(action: str, data: dict) -> tuple[int, dict]:
    """Route an action to its handler. Returns (http_status, response_body)."""
    if action not in ALLOWED_ACTIONS:
        return 403, {"error": f"Action '{action}' is not permitted."}
    handler = _handlers.get(action)
    if handler is None:
        return 501, {"error": f"Action '{action}' is allowed but has no handler registered."}
    try:
        result = handler(data)
        return 200, {"ok": True, "result": result}
    except Exception as e:
        return 500, {"error": str(e)}


# ── Built-in handlers ────────────────────────────────────────────────────────

def _handle_ping(_data: dict) -> dict:
    return {"pong": True, "ts": time.time(), "app": "llm_parametizer"}

register_handler("ping", _handle_ping)
register_handler("shield_heartbeat", lambda d: {"ack": True, "ts": time.time()})


# ── HTTP server ───────────────────────────────────────────────────────────────

class _IPCRequestHandler(BaseHTTPRequestHandler):
    """Handles POST /ipc with JSON body {timestamp, data: {action, ...}}."""

    def do_POST(self):
        if self.path != "/ipc":
            self._respond(404, {"error": "Not found. Use POST /ipc"})
            return

        # Read body
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._respond(400, {"error": f"Invalid JSON: {e}"})
            return

        # Validate envelope: {timestamp: number, data: object}
        timestamp = body.get("timestamp")
        data = body.get("data")
        if not isinstance(timestamp, (int, float)):
            self._respond(400, {"error": "Missing or invalid 'timestamp' (must be a number)."})
            return
        if not isinstance(data, dict):
            self._respond(400, {"error": "Missing or invalid 'data' (must be an object)."})
            return

        action = data.get("action", "")
        if not action:
            self._respond(400, {"error": "Missing 'action' in data."})
            return

        status, result = _dispatch(action, data)
        self._respond(status, result)

    def do_GET(self):
        """GET /ipc/status — health probe + safe status for peer detection."""
        if self.path in ("/ipc/status", "/ipc/ping"):
            body = {"ok": True, "app": "llm_parametizer", "ts": time.time()}
            if _status_provider is not None:
                try:
                    safe = _status_provider()
                    if isinstance(safe, dict):
                        body.update(safe)
                except Exception:
                    pass   # never let a status error break the health probe
            self._respond(200, body)
        else:
            self._respond(404, {"error": "Not found."})

    def do_OPTIONS(self):
        """CORS preflight for browser-based callers."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _respond(self, status: int, body: dict):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, _format, *args):
        # Silence default stderr logging
        pass


# ── Server lifecycle ──────────────────────────────────────────────────────────

_server: Optional[HTTPServer] = None
_thread: Optional[threading.Thread] = None


def start(port: int = DEFAULT_IPC_PORT) -> bool:
    """Start the IPC HTTP server on a daemon thread. Returns True on success."""
    global _server, _thread
    if _server is not None:
        return True  # already running
    try:
        _server = HTTPServer(("127.0.0.1", port), _IPCRequestHandler)
        _thread = threading.Thread(target=_server.serve_forever, daemon=True)
        _thread.start()
        return True
    except OSError:
        _server = None
        return False


def stop():
    """Shut down the IPC server."""
    global _server, _thread
    if _server is not None:
        _server.shutdown()
        _server = None
        _thread = None

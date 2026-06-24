"""
Phase 2 streaming integration test.

Spins up a tiny local HTTP server that emulates BOTH:
  - llama-server  /v1/chat/completions  (OpenAI SSE format)
  - Ollama        /api/chat             (newline-delimited JSON)

Verifies the engine and ChatSession parse real streamed bytes correctly.

Run:
    python tests/test_phase2_streaming.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_parametizer.chat.engine import Backend, ChatStats, stream_chat
from llm_parametizer.chat.session import ChatSession


TOKENS = ["Hello", ", ", "world", "!"]


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # silence

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        if self.path == "/v1/chat/completions":
            self._stream_openai()
        elif self.path == "/api/chat":
            self._stream_ollama()
        else:
            self.send_response(404)
            self.end_headers()

    def _stream_openai(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for tok in TOKENS:
            chunk = {"choices": [{"delta": {"content": tok}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_ollama(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for tok in TOKENS:
            obj = {"message": {"role": "assistant", "content": tok}, "done": False}
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()
        self.wfile.write((json.dumps({"done": True}) + "\n").encode())
        self.wfile.flush()


def _start_server() -> tuple[HTTPServer, int]:
    srv = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_stream_llamacpp_parses_tokens():
    srv, port = _start_server()
    try:
        out = []
        stats = None
        for item in stream_chat(Backend.LLAMACPP,
                                [{"role": "user", "content": "hi"}],
                                host="127.0.0.1", port=port):
            if isinstance(item, ChatStats):
                stats = item
            else:
                out.append(item)
        assert "".join(out) == "Hello, world!"
        assert stats is not None and stats.ok
        assert stats.tokens == len(TOKENS)
    finally:
        srv.shutdown()


def test_stream_ollama_parses_tokens():
    srv, port = _start_server()
    try:
        out = []
        for item in stream_chat(Backend.OLLAMA,
                                [{"role": "user", "content": "hi"}],
                                host="127.0.0.1", port=port, model="x"):
            if isinstance(item, ChatStats):
                assert item.ok
            else:
                out.append(item)
        assert "".join(out) == "Hello, world!"
    finally:
        srv.shutdown()


def test_session_full_streaming_roundtrip():
    srv, port = _start_server()
    try:
        tokens = []
        done = threading.Event()
        result = {}

        def on_done(full, stats):
            result["full"] = full
            result["stats"] = stats
            done.set()

        sess = ChatSession(
            on_token=tokens.append,
            on_done=on_done,
            on_error=lambda m: (result.update(error=m), done.set()),
        )
        sess.send("hi", Backend.LLAMACPP, host="127.0.0.1", port=port)
        assert done.wait(timeout=5)
        assert result.get("full") == "Hello, world!"
        assert "".join(tokens) == "Hello, world!"
        # Assistant reply stored in history
        assert sess.history.messages[-1].content == "Hello, world!"
        assert sess.is_busy is False
    finally:
        srv.shutdown()


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)

"""Local dashboard server (stdlib only).

    python -m lotka.server [--port 8765] [--host 127.0.0.1]

Endpoints
  GET  /                    dashboard (static/index.html)
  GET  /api/schema          parameter schema (names, defaults, ranges, descriptions)
  POST /api/reset           body: JSON params  -> new simulation, returns state
  GET  /api/step?n=K&since=T advance K ticks, returns grid + history rows after tick T
  GET  /api/state?since=T   current state without stepping
  GET  /api/export.csv      full metrics history as CSV
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .sim import METRIC_KEYS, Params, Simulation

STATIC = Path(__file__).parent / "static"
MAX_CELLS = 1_000_000


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.sim = Simulation(Params())

    def reset(self, d: dict) -> None:
        p = Params.from_dict(d)
        if p.width * p.height > MAX_CELLS:
            raise ValueError(f"grid too large (max {MAX_CELLS} cells)")
        with self.lock:
            self.sim = Simulation(p)

    def state(self, since: int = -1, steps: int = 0) -> dict:
        with self.lock:
            s = self.sim
            if steps > 0 and not s.extinct:
                s.step(steps)
            hist = [h for h in s.history if h["tick"] > since]
            return {
                "tick": s.tick,
                "width": s.W,
                "height": s.H,
                "cells": s.W * s.H,
                "fertile": int(s.fertile.sum()),
                "extinct": s.extinct,
                "params": s.p.to_dict(),
                "grid": base64.b64encode(s.render_codes().tobytes()).decode(),
                "history": hist,
            }

    def csv(self) -> str:
        with self.lock:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=METRIC_KEYS)
            w.writeheader()
            w.writerows(self.sim.history)
            return buf.getvalue()


APP = App()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser reloaded / navigated away mid-request

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/schema":
                self._json(Params.schema())
            elif u.path == "/api/step":
                n = max(0, min(int(q.get("n", 1)), 1000))
                self._json(APP.state(int(q.get("since", -1)), n))
            elif u.path == "/api/state":
                self._json(APP.state(int(q.get("since", -1))))
            elif u.path == "/api/export.csv":
                self._send(200, APP.csv().encode(), "text/csv")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if u.path == "/api/reset":
                APP.reset(body)
                self._json(APP.state())
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description="Lotka dashboard server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Lotka dashboard on http://{a.host}:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Public, simulation-only server (for exposure via Tailscale Funnel).

    python -m lotka.public [--port 8766] [--host 127.0.0.1]

Separation from the local dashboard (`lotka serve`, port 8765):
  * No optimisation at all: /evolve and every /api/evo/* route are absent;
    do_GET/do_POST are whitelists, so nothing here can start/stop processes.
  * Each visitor (cookie `lotka_sid`) gets a private Simulation, so visitors
    cannot reset each other or the owner's local dashboard.
  * Resource caps: grid size, ticks per request, number of live sessions,
    idle expiry and a global limit on concurrent simulation work.
"""

from __future__ import annotations

import argparse
import secrets
import threading
import time
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import server
from .sim import Params

MAX_CELLS = 300 * 300        # largest grid a visitor may create
MAX_STEPS = 50               # ticks per /api/step request (UI max is 50)
MAX_SESSIONS = 30            # live visitor simulations kept in memory
IDLE_SECONDS = 30 * 60       # drop a visitor's simulation after this long unused
CONCURRENT_WORK = 2          # simultaneous simulation computations (of all visitors)
MAX_BODY = 64 * 1024

_WORK = threading.BoundedSemaphore(CONCURRENT_WORK)


class PublicApp(server.App):
    def reset(self, d: dict) -> None:
        p = Params.from_dict(d)
        if p.width * p.height > MAX_CELLS:
            raise ValueError(f"grid too large for the public demo (max {MAX_CELLS} cells, e.g. 300x300)")
        with self.lock:
            self.sim = server.Simulation(p)


class Sessions:
    def __init__(self):
        self.lock = threading.Lock()
        self.apps: dict[str, list] = {}   # sid -> [PublicApp, last_seen]

    def get(self, sid: str | None) -> tuple[str, PublicApp, bool]:
        now = time.monotonic()
        with self.lock:
            for k in [k for k, (_, t) in self.apps.items() if now - t > IDLE_SECONDS]:
                del self.apps[k]
            if sid and sid in self.apps:
                self.apps[sid][1] = now
                return sid, self.apps[sid][0], False
            while len(self.apps) >= MAX_SESSIONS:   # evict least recently used
                del self.apps[min(self.apps, key=lambda k: self.apps[k][1])]
            sid = secrets.token_urlsafe(18)
            app = PublicApp()
            self.apps[sid] = [app, now]
            return sid, app, True


SESSIONS = Sessions()


def _index_html() -> bytes:
    html = (server.STATIC / "index.html").read_text()
    # hide the link to the (local-only) optimisation command center
    out = "\n".join(l for l in html.splitlines() if 'href="/evolve"' not in l)
    return out.encode()


class PublicHandler(server.Handler):
    _cookie: str | None = None

    def _send(self, code: int, body: bytes, ctype: str):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if self._cookie:
                self.send_header("Set-Cookie", self._cookie)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _app(self) -> PublicApp:
        c = SimpleCookie(self.headers.get("Cookie", ""))
        sid = c["lotka_sid"].value if "lotka_sid" in c else None
        sid, app, new = SESSIONS.get(sid)
        if new:
            self._cookie = f"lotka_sid={sid}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age={IDLE_SECONDS}"
        return app

    def _work(self, fn):
        if not _WORK.acquire(timeout=10):
            return self._json({"error": "server busy, try again"}, 503)
        try:
            return self._json(fn())
        finally:
            _WORK.release()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._app()
                self._send(200, _index_html(), "text/html; charset=utf-8")
            elif u.path == "/api/schema":
                self._json(Params.schema())
            elif u.path == "/api/presets":
                self._json(server.all_presets())
            elif u.path == "/api/step":
                app, n = self._app(), max(0, min(int(q.get("n", 1)), MAX_STEPS))
                self._work(lambda: app.state(int(q.get("since", -1)), n))
            elif u.path == "/api/state":
                app = self._app()
                self._work(lambda: app.state(int(q.get("since", -1))))
            elif u.path == "/api/lv":
                app = self._app()
                self._work(lambda: app.lv(q.get("model", "rm"), int(q.get("fit_start", 0)),
                                          min(int(q.get("extend", 0)), 5000)))
            elif u.path == "/api/export.csv":
                self._send(200, self._app().csv().encode(), "text/csv")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self._json({"error": "request too large"}, 413)
            if u.path != "/api/reset":
                return self._send(404, b"not found", "text/plain")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json({"error": "expected application/json"}, 415)
            body = server.json.loads(self.rfile.read(length) or b"{}")
            app = self._app()
            app.reset(body)
            self._work(app.state)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description="Lotka public simulation-only server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), PublicHandler)
    srv.daemon_threads = True
    print(f"Lotka public (simulation only) on http://{a.host}:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

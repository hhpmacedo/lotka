"""Command center backend: launch, monitor, stop and resume evolution runs.

Runs live in runs/evolve/<name>/. Each run is a separate `python -m lotka evolve` process in its
own session, so it keeps going if the dashboard server restarts. The run writes status.json;
stdout goes to log.txt.
"""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .evolve import DEFAULT_GENES, METRIC_NAMES, OBJECTIVES
from .sim import Params

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "evolve"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TYPES = {f["name"]: f["type"] for f in Params.schema()}


def _run_dir(name: str) -> Path:
    if not NAME_RE.match(name or ""):
        raise ValueError("run name: letters, digits, _ and - only (max 64)")
    return RUNS / name


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:  # zombie check
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def meta() -> dict:
    return {
        "objectives": {k: {"desc": o.desc, "args": o.args} for k, o in OBJECTIVES.items()},
        "genes": {k: {"lo": lo, "hi": hi, "type": _TYPES[k]} for k, (lo, hi) in DEFAULT_GENES.items()},
        "params": Params.schema(),
        "metrics": METRIC_NAMES,
        "cpus": os.cpu_count() or 1,
        "defaults": {"pop": 48, "gens": 30, "seeds": 3, "ticks": 1500, "size": 100,
                     "workers": os.cpu_count() or 1, "mut_sigma": 0.12, "validate_seeds": 10},
    }


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _status(d: Path) -> dict:
    st = _read_json(d / "status.json") or {}
    cfg = _read_json(d / "config.json") or {}
    state = st.get("state")
    if state == "running" and not _alive(st.get("pid")):
        state = "died"
    if not state:  # runs from before status.json existed
        if (d / "best.json").exists():
            state = "finished"
        elif (d / "checkpoint.json").exists() and time.time() - (d / "checkpoint.json").stat().st_mtime < 600:
            state = "running"
        else:
            state = "stopped"
    gens_rows = _generations(d)
    last = gens_rows[-1] if gens_rows else {}
    best = _read_json(d / "best.json")
    return {
        "name": d.name, "state": state, "phase": st.get("phase"), "error": st.get("error"),
        "objective": cfg.get("objective"), "args": cfg.get("args", {}),
        "gen": int(last.get("gen", st.get("gen", 0)) or 0), "gens": cfg.get("gens"),
        "pop": cfg.get("pop"), "evals": int(last.get("evals", 0) or 0),
        "best": float(last["best"]) if last.get("best") else None,
        "coexist_frac": float(last["coexist_frac"]) if last.get("coexist_frac") else None,
        "archive_cells": int(last.get("archive_cells", 0) or 0),
        "validated": best.get("validated_fitness") if best else None,
        "started": st.get("started") or (d / "config.json").stat().st_mtime if (d / "config.json").exists() else None,
        "updated": (d / "generations.csv").stat().st_mtime if (d / "generations.csv").exists() else None,
        "workers": cfg.get("workers"),
    }


def _generations(d: Path) -> list[dict]:
    p = d / "generations.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def list_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    runs = [_status(d) for d in RUNS.iterdir() if d.is_dir() and (d / "config.json").exists()]
    return sorted(runs, key=lambda r: -(r["started"] or 0))


def run_detail(name: str) -> dict:
    d = _run_dir(name)
    if not (d / "config.json").exists():
        raise ValueError(f"no run named {name}")
    log = ""
    if (d / "log.txt").exists():
        log = "\n".join((d / "log.txt").read_text(errors="replace").splitlines()[-80:])
    archive = _read_json(d / "archive.json")
    if archive:
        archive = {"axes": archive["axes"],
                   "cells": {k: {"fitness": v["fitness"], "metrics": v["metrics"], "params": v["params"]}
                             for k, v in archive["cells"].items()}}
    return {"status": _status(d), "config": _read_json(d / "config.json"),
            "generations": _generations(d), "best": _read_json(d / "best.json"),
            "archive": archive, "log": log}


def busy_workers() -> int:
    # runs started before 'workers' was recorded used all cores
    return sum(r.get("workers") or (os.cpu_count() or 1) for r in list_runs() if r["state"] == "running")


def _spawn(d: Path, args: list[str]):
    log = (d / "log.txt").open("a")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(args)}\n")
    log.flush()
    proc = subprocess.Popen([sys.executable, "-u", "-m", "lotka", "evolve", *args], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    # record pid immediately so the UI shows it as running before the first generation
    (d / "status.json").write_text(json.dumps({"state": "running", "pid": proc.pid, "phase": "starting",
                                               "started": time.time(), "updated": time.time()}))
    return proc.pid


def start(body: dict) -> dict:
    name = body.get("name") or f"{body.get('objective', 'run')}_{time.strftime('%Y%m%d_%H%M%S')}"
    d = _run_dir(name)
    if d.exists():
        raise ValueError(f"a run named {name} already exists")
    obj = body.get("objective")
    if obj not in OBJECTIVES:
        raise ValueError(f"unknown objective {obj}")
    args = ["--objective", obj, "--out", str(d)]
    for k in ("pop", "gens", "seeds", "ticks", "size", "workers", "validate_seeds"):
        if body.get(k) not in (None, ""):
            args += [f"--{k.replace('_', '-')}", str(int(body[k]))]
    if body.get("mut_sigma") not in (None, ""):
        args += ["--mut-sigma", str(float(body["mut_sigma"]))]
    for k, v in (body.get("args") or {}).items():
        if str(v).strip():
            if not re.match(r"^[a-z_]+$", k):
                raise ValueError(f"bad objective argument {k}")
            args += ["--arg", f"{k}={v}"]
    genes = body.get("genes") or {}
    if genes:
        for k in genes:
            if k not in _TYPES:
                raise ValueError(f"unknown gene {k}")
        args += ["--genes", ",".join(genes)]
        for k, (lo, hi) in genes.items():
            args += ["--gene", f"{k}={float(lo)}:{float(hi)}"]
    for k, v in (body.get("fixed") or {}).items():
        if k not in _TYPES:
            raise ValueError(f"unknown parameter {k}")
        args += ["--fix", f"{k}={v}"]
    d.mkdir(parents=True)
    pid = _spawn(d, args)
    # remember workers for the CPU budget display
    return {"name": name, "pid": pid, "args": args}


def stop(name: str) -> dict:
    d = _run_dir(name)
    st = _read_json(d / "status.json") or {}
    pid = st.get("pid")
    if not _alive(pid):
        raise ValueError("run is not running")
    os.kill(pid, signal.SIGTERM)  # evolve.py stops cleanly and checkpoints
    return {"name": name, "signalled": pid}


def resume(name: str, gens: int, workers: int | None) -> dict:
    d = _run_dir(name)
    st = _status(d)
    if st["state"] == "running":
        raise ValueError("run is already running")
    args = ["--resume", str(d), "--gens", str(int(gens))]
    if workers:
        args += ["--workers", str(int(workers))]
    return {"name": name, "pid": _spawn(d, args)}


def file_path(name: str, f: str) -> Path:
    if f not in ("evals.csv", "best.json", "evolution.png", "generations.csv", "archive.json", "log.txt", "config.json"):
        raise ValueError("file not allowed")
    p = _run_dir(name) / f
    if not p.exists():
        raise ValueError(f"{f} not available yet")
    return p

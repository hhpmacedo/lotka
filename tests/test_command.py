import json
import time

import pytest

from lotka import command


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "RUNS", tmp_path)
    return tmp_path


def wait_state(name, states, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = command.list_runs()
        s = next((r for r in st if r["name"] == name), None)
        if s and s["state"] in states:
            return s
        time.sleep(0.5)
    raise AssertionError(f"{name} never reached {states}: {s}")


def test_name_validation(runs_dir):
    for bad in ("../x", "a b", "", "x" * 65, "a/b"):
        with pytest.raises(ValueError):
            command._run_dir(bad)


def test_start_stop_resume_finish(runs_dir):
    body = {"name": "t1", "objective": "coexistence", "pop": 4, "gens": 30, "seeds": 1, "ticks": 60,
            "size": 20, "workers": 1, "validate_seeds": 1,
            "genes": {"predator_starve_time": [2, 30], "prey_repro_prob": [0.01, 0.3]},
            "fixed": {"veg_regrow_time": 15}}
    r = command.start(body)
    assert "--genes" in r["args"] and "--fix" in r["args"]
    with pytest.raises(ValueError):
        command.start(body)  # duplicate name
    wait_state("t1", {"running"})
    # wait for at least one generation, then stop
    t0 = time.time()
    while not (runs_dir / "t1" / "checkpoint.json").exists() and time.time() - t0 < 60:
        time.sleep(0.3)
    command.stop("t1")
    s = wait_state("t1", {"stopped"})
    assert s["gen"] < 30
    cfg = json.loads((runs_dir / "t1" / "config.json").read_text())
    assert set(cfg["genes"]) == {"predator_starve_time", "prey_repro_prob"}
    assert cfg["fixed"]["veg_regrow_time"] == 15
    command.resume("t1", gens=s["gen"] + 1, workers=1)
    s = wait_state("t1", {"finished"}, timeout=180)
    d = command.run_detail("t1")
    assert d["best"]["params"]["veg_regrow_time"] == 15
    assert "resume" in d["log"]


def test_rejects_bad_input(runs_dir):
    with pytest.raises(ValueError):
        command.start({"name": "x", "objective": "nope"})
    with pytest.raises(ValueError):
        command.start({"name": "y", "objective": "coexistence", "genes": {"rm -rf": [0, 1]}})
    with pytest.raises(ValueError):
        command.file_path("x", "../../etc/passwd")

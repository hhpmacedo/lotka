"""Lotka command line.

Simulation level:
  python -m lotka serve     [--port 8765]          live dashboard
  python -m lotka batch     [--preset ...] [--sweep ...]   headless runs / parameter sweeps
  python -m lotka compare   [--preset ...]          ABM vs Lotka-Volterra report

Optimisation level:
  python -m lotka evolve    --objective coexistence|target_cycles|msy|amplitude|lv_match|custom|qd
  python -m lotka evolve    --list
"""

import sys


def main():
    cmds = {"serve": "server", "batch": "batch", "compare": "compare", "evolve": "evolve"}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    mod = __import__(f"lotka.{cmds[sys.argv[1]]}", fromlist=["main"])
    sys.argv = [f"lotka {sys.argv[1]}"] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Full-fidelity experiment batch for the paper. Safe to run detached:
#   setsid nohup bash paper/run_all.sh > paper/run_all.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "### start $(date '+%F %T')"
for e in E3 E4 E6a E6b E6c E6d E8; do
  $PY -u -m experiments.exp "$e" || echo "!!! $e failed (exit $?)"
done
if [ ! -f runs/evolve/yield_v1/best.json ]; then
  echo "### yield_v1 $(date '+%F %T')"
  $PY -u -m lotka evolve --objective yield --arg species=prey \
    --genes harvest_mode,harvest_effort,harvest_quota,harvest_setpoint,harvest_gain \
    --gene harvest_mode=1:3 --gene harvest_effort=0:0.1 --gene harvest_quota=0:40 \
    --gene harvest_setpoint=0:1500 --gene harvest_gain=0:1 \
    --fix harvest_species=1 --pop 32 --gens 20 --ticks 1500 --size 100 --seeds 3 \
    --out runs/evolve/yield_v1 || echo "!!! yield_v1 failed"
fi
$PY -u -m experiments.exp E5 || echo "!!! E5 failed"
echo "### figures $(date '+%F %T')"
$PY -u -m experiments.figures || echo "!!! figures failed"
echo "### done $(date '+%F %T')"

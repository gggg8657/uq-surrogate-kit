#!/usr/bin/env bash
# H9 must not overlap the repeatability probe on GPU 3: two timing jobs on one
# device measure each other's contention, not their own work.
set -u
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
cd "$(dirname "$0")/.."
while pgrep -f "probe_solver_repeat.py" >/dev/null; do sleep 10; done
echo "probe finished at $(date -Is)"
CUDA_VISIBLE_DEVICES=3 $PY scripts/bench_fair.py \
    --ckpt runs/u0/best.pt --uq-source het --task darcy \
    --batches 1 64 --trials 5 --out runs/bench_fair.json \
    > logs/bench_fair.log 2>&1 || echo "bench_fair FAILED rc=$?"
echo "H9_DONE $(date -Is)"

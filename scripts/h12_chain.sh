#!/usr/bin/env bash
# H12: re-run clause 2 with the packed-weight spectral path in place.
#
# EVERY solver flag is identical to the H11 run that produced the 21/24 verdict
# (`runs/bench_fair.json`: --fast-apply, --precond discrete, per-sample
# check_every from [1,10,50,100], 24 samples, 2 trials/cell, 5 trials, 10
# iters). `--precond continuous discrete` is what H11 ran: dropping
# `continuous` changes nothing about the chosen denominator (the discrete
# symbol is faster and wins the `min_s` selection either way) but it removes
# the `check_every=1` row the subsidized comparator is quoted against, which
# is how the first H12 launch crashed. Only the surrogate changed, and it
# changed bit-identically -- `packed_weight_gate` in the JSON is the proof.
set -u
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=2 $PY scripts/bench_fair.py \
    --ckpt runs/u0/best.pt --uq-source het --task darcy \
    --batches 1 64 --trials 5 --out runs/bench_fair_h12.json \
    --fast-apply --precond continuous discrete \
    > logs/bench_fair_h12.log 2>&1 || echo "bench_fair_h12 FAILED rc=$?"
echo "H12_BENCH_DONE $(date -Is)"

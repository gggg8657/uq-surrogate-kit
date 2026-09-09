#!/usr/bin/env bash
# Every timing measurement, re-run with the clock ramp in place.
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python
export CUDA_VISIBLE_DEVICES=3
CK=(runs/m*/best.pt)
for s in bench_noise bench_speedup bench_isoaccuracy ablate_members; do
  echo "=== $s $(date -Is)"
  $PY "scripts/$s.py" --ckpts "${CK[@]}" > "logs/$s.log" 2>&1 || echo "!!! $s failed"
  tail -3 "logs/$s.log"
done
$PY scripts/report.py
echo "=== timing rerun done $(date -Is)"

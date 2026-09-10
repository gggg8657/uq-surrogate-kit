#!/usr/bin/env bash
# H22: the PDE residual as a width feature. 8 seeds.
#
# QUEUES rather than oversubscribes. The lease is devices 2 and 3 and both were
# busy with this track's own H21 clause-3 chain and the H17 cost benchmark when
# this was written, so the chain waits for its device to fall idle before the
# first fit and does not start a second process alongside them.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-2}

wait_for_device() {
  # idle = <10% utilisation on three consecutive samples 20 s apart
  local hits=0
  while [ "$hits" -lt 3 ]; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$DEV")
    if [ "$u" -lt 10 ]; then hits=$((hits+1)); else hits=0; fi
    sleep 20
  done
  echo "device $DEV idle; starting"
}

wait_for_device
for k in 0 1 2 3 4 5 6 7; do
  out=runs/scalerf_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --residual-features \
      --out "$out" || { echo "FAILED seed $k"; exit 1; }
done
echo "H22 chain done: $(ls runs/scalerf_u*_het.json | wc -l)/8"

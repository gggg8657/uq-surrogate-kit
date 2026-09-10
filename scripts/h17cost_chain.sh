#!/usr/bin/env bash
# H17's cost to clause 2, measured rather than extrapolated.
#
# The wrapper's overhead was previously quoted from the eager
# `predict_shard_single` path and then divided into a CUDA-graph speedup --
# arithmetic across two protocols, withdrawn, and the cell has read
# `[not measured]` since. It could not be measured because
# `bench_fair.py --equivariant` died in graph capture: `sample_scale` gathered
# its channels with `a[:, list(ch)]`, and advanced indexing builds its index
# tensor on the host, which is illegal while a stream is capturing. Fixed by
# slicing (bit-identical, pinned by tests/test_equivar.py).
#
# BOTH arms run here, on ONE device, back to back. The published 117.0x came
# off GPU 2 and this runs on whichever device is free; a speedup delta measured
# across two devices is not a delta, so the baseline is re-measured beside the
# equivariant arm rather than reused. Every other flag is byte-identical to
# scripts/h12_chain.sh.
set -u
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-3}
COMMON="--ckpt runs/u0/best.pt --uq-source het --task darcy \
        --batches 1 64 --trials 5 --fast-apply --precond continuous discrete"
for arm in base eq; do
  out=runs/bench_fair_h17_${arm}.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  flag=""; [ "$arm" = "eq" ] && flag="--equivariant"
  echo "=== arm $arm on cuda:$DEV $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY scripts/bench_fair.py $COMMON $flag \
      --out "$out" > logs/bench_fair_h17_${arm}.log 2>&1 \
      || { echo "FAILED arm $arm rc=$?"; tail -5 logs/bench_fair_h17_${arm}.log; exit 1; }
  echo "wrote $out"
done
echo "H17COST_DONE $(date -Is)"

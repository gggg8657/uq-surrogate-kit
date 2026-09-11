#!/usr/bin/env bash
# H23: the clause-2 cost of the H17 equivariance wrapper, measured under CUDA
# graph replay -- the protocol the 117.0x clause-2 figure was measured on.
#
# This was `[not measured]` in runs/equivar_cost.json for a real reason: capture
# aborted with "operation not permitted when stream is capturing" because
# `sample_scale` gathered channels with `a[:, list(ch)]`, which builds its index
# tensor on the host. That is fixed (slices now) and pinned by
# tests/test_equivar.py::test_scale_is_cuda_graph_capturable, so the measurement
# is possible for the first time.
#
# Timing job: runs only on an idle leased GPU, never alongside another job.
set -u
cd "$(dirname "$0")/.."
for ARM in base eq; do
  flag=""; [ "$ARM" = eq ] && flag="--equivariant"
  out=runs/bench_fair_h23_${ARM}.json
  [ -s "$out" ] && { echo "skip $out"; continue; }
  ./scripts/run_when_gpu_free.sh 2,3 -- \
      ~/miniforge3/envs/pdeno/bin/python -u scripts/bench_fair.py \
      $flag --out "$out" || exit 1
done
echo "H23 done"

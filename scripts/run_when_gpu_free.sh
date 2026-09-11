#!/usr/bin/env bash
# Wait until a leased GPU has no compute process, then run the command on it.
#
# This exists because of a specific failure mode this repo has already hit
# twice: a timing number measured while something else shared the device. The
# clock-ramp artefact (40.8x -> 23.5x) and the +-36% run-to-run spread were
# both contention/thermal effects, and a contended benchmark does not announce
# itself -- it just reports a slower surrogate or a faster solver.
#
#   ./scripts/run_when_gpu_free.sh 2,3 -- python scripts/bench_fair.py ...
#
# Coverage and fitting jobs do not need this; timing jobs do. Polls every 60 s
# and prints what it is waiting for, so a stalled queue is visible rather than
# silent.
set -u
LEASE="${1:?usage: run_when_gpu_free.sh <comma-separated gpu ids> -- cmd...}"
shift
[ "${1:-}" = "--" ] && shift

free_gpu() {
  for g in ${LEASE//,/ }; do
    uuid=$(nvidia-smi --id="$g" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
    [ -z "$uuid" ] && continue
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | grep -c "$uuid")
    if [ "$n" -eq 0 ]; then echo "$g"; return 0; fi
  done
  return 1
}

waited=0
while true; do
  if g=$(free_gpu); then
    echo "[gpu-queue] GPU $g is idle after ${waited}s; starting: $*"
    CUDA_VISIBLE_DEVICES="$g" exec "$@"
  fi
  [ $((waited % 300)) -eq 0 ] && \
    echo "[gpu-queue] waiting ${waited}s for one of GPU $LEASE to go idle" >&2
  sleep 60; waited=$((waited + 60))
done

#!/usr/bin/env bash
# Block until no pdeno python process is executing one of the timing scripts.
#
# WHY THIS IS NOT `pgrep -f scripts/bench_fair.py`: that pattern matches any
# command line CONTAINING the string. A shell command that greps for the name,
# an editor with the file open, and the agent process -- whose argv carries the
# whole brief -- all match it. Measured twice today: the H25 chain logged
# "waiting for the lease (600s)" and the H27 chain blocked at 0s, both against a
# `zsh -c` whose argv merely mentioned the filename, on an idle GPU. The same
# class of bug then made `pkill -f h27_chain.sh` kill the shell that ran it.
# Match the interpreter and the script together, and exclude self and parent.
set -u
PAT='pdeno/bin/python[^ ]*( +-[^ ]+)* +[^ ]*scripts/(bench_fair|bench_speedup|rerun_timing)\.py'
waited=0
while :; do
  hits=$(pgrep -a -f "$PAT" 2>/dev/null | awk -v me=$$ -v pp="$PPID" '$1!=me && $1!=pp')
  [ -z "$hits" ] && break
  if [ $((waited % 300)) -eq 0 ]; then
    echo "waiting for the timing lease (${waited}s); held by:"
    echo "$hits" | cut -c1-140 | sed 's/^/    /'
  fi
  sleep 30; waited=$((waited + 30))
done
[ "$waited" -gt 0 ] && echo "lease released after ${waited}s"
exit 0

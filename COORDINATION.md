# Two instances are running this brief in this repository

Recorded because it has already cost real work, twice, in both directions.

Observed PIDs: **3452163** (started 2026-09-11 03:30:20) and **3643314**
(03:40:31). Both are running the same weekend brief with the same queue, so
both independently pick the same next experiment and the same obvious filenames.

## What has already gone wrong

| when | what | cost |
|---|---|---|
| 04:08 | one instance edited `scripts/eval_label_budget.py` (+152 lines, arms B and C) while the other's 8-seed chain was running against it | the peer withdrew a completed 8-seed run in `decef5a`, correctly, because the code its seeds ran against no longer existed |
| 04:1x | both instances created `scripts/h30_chain.sh`; the peer's cleanup deleted it while the other's chain was on seed 6 | the chain survived only because bash held an unlinked file descriptor |
| earlier | a lease gate of the form `pgrep -f "scripts/<name>.py"` matched its own `sh -c` argv | one job spun **19 hours** on an idle GPU and produced nothing |

## Rules that would have prevented each of them

1. **Namespace anything you create.** Chain scripts, run JSONs and logs get a
   suffix that identifies the hypothesis *and* is unlikely to collide:
   `scripts/h30b_chain.sh`, `runs/label_budget_u*_het.json`. Do not take a bare
   `hNN_chain.sh` name — the other instance is numbering hypotheses from the
   same critique log and will reach the same NN.
2. **Never edit a script while a chain is reading it.** Check first:
   `pgrep -af "scripts/<the-script>"`. If a chain is live, copy the script to a
   new name, edit the copy, and point a new chain at it. The run JSON should
   record which file produced it.
3. **Do not delete another instance's files.** `git status` will show untracked
   runs you did not create. Leave them. If a name collides, rename *yours*.
4. **Use `scripts/lease_wait.sh`, never a bare `pgrep -f <filename>`.** It
   matches the interpreter and the script together and excludes self and
   parent, and `tests/test_scale_target.py` pins that it does not match a
   command line which merely names the scripts it waits on.
5. **`critique_log.md` is append-only.** Both instances append; neither
   rewrites. This is the one file that has survived the collisions intact, and
   it is why the peer's withdrawal and this note can coexist.
6. **GPU lease is devices 2 and 3 for this track.** Check
   `nvidia-smi` before launching; a transient allocation by the other instance
   is not a reason to take device 0 or 1.

## The substantive lesson, which is not about process

Both instances independently ran the same label-budget experiment and got the
same `49/49 shards in band at k=9`. One withdrew it for an operational reason
(unstable code) and one explained it: split conformal's mean coverage is
`(k+1-l)/(k+1)` with `l = floor((k+1)*alpha)`, which is `9/10 = 0.9000` exactly
at k=9, so "the mean is in band" is integer arithmetic rather than a result.
Duplicated effort found the defect twice and the explanation once. Reading
`critique_log.md` before choosing a hypothesis is cheaper than rediscovering it.

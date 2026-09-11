"""The s*-target machinery, pinned where it can be pinned exactly.

H27 concludes that no deployment-observable scalar reaches the coverage band,
and the whole conclusion rests on `s*` being the *right* target. Three things
could make it wrong in a way that would look like a finding:

* coverage might not be monotone in the applied scale, in which case the
  bisection converges to an arbitrary point rather than to the crossing;
* the additive floor might be scaled along with sigma, which would make `s*`
  a statement about a construction this kit does not use;
* an aggregator might quote an oracle target as a method result.

Each is checked here as an identity. The fourth test pins the exact binomial
arithmetic behind the claim that our own strict clause cannot be passed by a
perfectly calibrated method, because that number goes in the KPI table and it
came from a critic rather than from a run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uqkit.conformal import _floor, score_field_max  # noqa: E402


def test_coverage_is_monotone_in_the_scale():
    """The bisection's only assumption, on random data, at both extremes."""
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(64, 1, 16, 16, generator=g)
    truth = pred + 0.3 * torch.randn(64, 1, 16, 16, generator=g)
    sigma = torch.rand(64, 1, 16, 16, generator=g).clamp_min(1e-3)
    med = float(sigma.median())
    q = float(np.quantile(
        score_field_max(pred, sigma, truth, med=med).numpy(), 0.9))
    covs = []
    for s in np.geomspace(1e-4, 1e4, 25):
        sc = score_field_max(pred, sigma * float(s), truth, med=med).numpy()
        covs.append(float((sc <= q).mean()))
    d = np.diff(covs)
    assert (d >= -1e-12).all(), f"coverage not monotone in the scale: {covs}"
    assert covs[0] < 0.90 < covs[-1], \
        f"the crossing is not bracketed by [1e-4, 1e4]: {covs[0]}, {covs[-1]}"
    print(f"ok  coverage monotone in the scale over 8 decades, "
          f"{covs[0]:.3f} -> {covs[-1]:.3f}, 0.90 bracketed")


def test_the_floor_does_not_move_with_the_scale():
    """`med=MED` must keep the floor at its frozen calibration value.

    If the floor scaled with sigma, `s*` would describe a purely multiplicative
    interval and this kit's interval is `s*sigma + 0.05*med`. The difference is
    what makes the H26 floor-share measurement meaningful, so it is pinned.
    """
    sigma = torch.full((4, 1, 8, 8), 2.0)
    med = 10.0
    for s in (0.1, 1.0, 7.0):
        got = _floor(sigma * s, 0.05, med)
        want = sigma * s + 0.05 * med
        assert torch.equal(got, want), f"floor moved at s={s}"
    # and the wrong call -- passing med/s -- is genuinely different, so the
    # test would have caught the bug it exists to prevent
    assert not torch.equal(_floor(sigma * 7.0, 0.05, med / 7.0),
                           sigma * 7.0 + 0.05 * med)
    print("ok  the additive floor stays frozen under a sigma rescale, and the "
          "med/s variant is measurably different")


def test_aggregators_refuse_to_quote_an_oracle_as_a_result():
    """agg_h27 must reject a run not marked `clause_eligible: false`."""
    src = (ROOT / "runs").glob("starget_u*_het.json")
    files = sorted(src)
    if not files:
        print("skip no H27 runs on disk")
        return
    for f in files:
        assert json.loads(f.read_text())["clause_eligible"] is False, \
            f"{f.name} must carry clause_eligible: false"
    tmp = ROOT / "runs" / "starget_uTESTBAD_het.json"
    d = json.loads(files[0].read_text())
    d["clause_eligible"] = True
    tmp.write_text(json.dumps(d))
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts/agg_h27.py")],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0, "agg_h27 accepted a clause-eligible s* run"
        assert "clause_eligible" in (r.stdout + r.stderr)
    finally:
        tmp.unlink()
    print(f"ok  {len(files)} s* runs marked clause_eligible: false, and "
          f"agg_h27 exits nonzero on one that is not")


def test_the_strict_clause_ceiling_is_what_we_published():
    """A perfectly calibrated method cannot pass our own strict reading.

    n=512 fields per shard, true coverage exactly 0.90. These four numbers are
    quoted in the KPI table and in paper_draft.md, and they came from a critic's
    argument, so the arithmetic is pinned rather than trusted.
    """
    from scipy.stats import binom
    n, p = 512, 0.90
    lo = int(np.ceil(0.88 * n))
    hi = int(np.floor(0.92 * n))
    per_shard = binom.cdf(hi, n, p) - binom.cdf(lo - 1, n, p)
    assert abs(per_shard - 0.8787) < 5e-5, per_shard
    assert abs(24 * per_shard - 21.09) < 5e-3, 24 * per_shard
    p24 = per_shard ** 24
    assert abs(p24 - 0.0449) < 5e-4, p24
    p22 = sum(binom.pmf(k, 24, per_shard) for k in range(22, 25))
    assert abs(p22 - 0.4299) < 5e-4, p22
    print(f"ok  a perfect method scores {24*per_shard:.2f}/24 in expectation, "
          f"24/24 with probability {p24:.4f}, >=22/24 with {p22:.4f}")


def test_lease_gate_does_not_match_its_own_name():
    """The bug that cost H25 ten minutes: `pgrep -f <file>` matches the shell.

    Run the gate from a shell whose command line contains the exact script
    names it waits on. It must return immediately.
    """
    sh = ROOT / "scripts" / "lease_wait.sh"
    if not sh.is_file():
        print("skip scripts/lease_wait.sh absent")
        return
    r = subprocess.run(
        ["bash", "-c",
         f"# scripts/bench_fair.py scripts/bench_speedup.py\n"
         f"timeout 20 {sh}"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (
        f"lease gate blocked on its own name (exit {r.returncode}): "
        f"{r.stdout[-300:]}")
    assert "waiting" not in r.stdout, f"gate waited: {r.stdout[:200]}"
    print("ok  the lease gate ignores a command line that merely names the "
          "scripts it waits on")


if __name__ == "__main__":
    test_coverage_is_monotone_in_the_scale()
    test_the_floor_does_not_move_with_the_scale()
    test_aggregators_refuse_to_quote_an_oracle_as_a_result()
    test_the_strict_clause_ceiling_is_what_we_published()
    test_lease_gate_does_not_match_its_own_name()
    print("all s*-target tests passed")

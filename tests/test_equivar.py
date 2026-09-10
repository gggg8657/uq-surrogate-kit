"""The H17 scale wrapper, checked as an identity rather than by its effect.

`F_eq(a) = s(a) * F(a/s(a))` with `s(c*a) = c*s(a)` is exactly equivariant for
*any* `F`, so the property can be pinned without reference to the network --
which is the point: if the identity holds and `darcy_amp2` still improves, the
improvement came from somewhere else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uqkit.equivar import (LINEAR_CHANNELS, predict_equivariant,  # noqa: E402
                           reference_scale, rescale_inputs, sample_scale)


def _fake_predict(blob):
    """A deliberately NON-equivariant stand-in for the network."""
    a = blob["a"]
    m = torch.tanh(a[:, 0:1]) + 0.3 * a[:, 0:1] ** 2
    return m, 0.1 * m.abs() + 0.01, torch.zeros_like(m), a


def test_scale_is_homogeneous_of_degree_one():
    a = torch.randn(16, 2, 8, 8)
    for fam in LINEAR_CHANNELS:
        s1 = sample_scale(a, fam, ref=1.0)
        for c in (0.5, 2.0, 7.5):
            b = a.clone()
            for ch in LINEAR_CHANNELS[fam]:
                b[:, ch] = b[:, ch] * c
            s2 = sample_scale(b, fam, ref=1.0)
            assert torch.allclose(s2, c * s1, rtol=1e-6), (fam, c)


def test_nonlinear_channels_are_untouched():
    """Darcy's log-permeability must survive the wrapper unchanged."""
    a = torch.randn(8, 2, 8, 8)
    s = sample_scale(a, "darcy", ref=1.0)
    out = rescale_inputs(a, "darcy", s)
    assert torch.equal(out[:, 0], a[:, 0]), "log-permeability was rescaled"
    assert not torch.equal(out[:, 1], a[:, 1]), "the source was not rescaled"


def test_wrapper_is_exactly_equivariant_despite_a_nonlinear_predictor():
    a = torch.randn(12, 2, 8, 8)
    for fam in ("poisson", "darcy"):
        ref = reference_scale(a, fam)
        blob = {"a": a, "task": fam}
        m1, s1, _t, _a = predict_equivariant(_fake_predict, blob, fam, ref)
        for c in (0.25, 3.0, 50.0):
            b = a.clone()
            for ch in LINEAR_CHANNELS[fam]:
                b[:, ch] = b[:, ch] * c
            m2, s2, _t2, _a2 = predict_equivariant(
                _fake_predict, {"a": b, "task": fam}, fam, ref)
            assert torch.allclose(m2, c * m1, rtol=1e-5, atol=1e-6), (fam, c)
            # sigma must scale with the mean or the interval no longer belongs
            # to the prediction it is attached to
            assert torch.allclose(s2, c * s1, rtol=1e-5, atol=1e-6), (fam, c)
        # the bare predictor must NOT have this property, or the test is vacuous
        m_bare, _, _, _ = _fake_predict({"a": a, "task": fam})
        b = a.clone()
        for ch in LINEAR_CHANNELS[fam]:
            b[:, ch] = b[:, ch] * 3.0
        m_bare3, _, _, _ = _fake_predict({"a": b, "task": fam})
        assert not torch.allclose(m_bare3, 3.0 * m_bare, rtol=1e-3)


def test_wrapper_is_near_identity_at_the_reference_scale():
    """In distribution s ~ 1, so the wrapper must barely move the prediction."""
    a = torch.randn(64, 2, 8, 8)
    ref = reference_scale(a, "poisson")
    s = sample_scale(a, "poisson", ref)
    assert abs(float(s.median()) - 1.0) < 1e-6
    m_eq, _, _, _ = predict_equivariant(_fake_predict, {"a": a,
                                                        "task": "poisson"},
                                        "poisson", ref)
    m_raw, _, _, _ = _fake_predict({"a": a, "task": "poisson"})
    # not identical -- F is nonlinear and s varies per sample -- but the median
    # relative change must be small, or "near-identity in distribution" is false
    rel = ((m_eq - m_raw).flatten(1).norm(dim=1)
           / m_raw.flatten(1).norm(dim=1).clamp_min(1e-12))
    assert float(rel.median()) < 0.25, float(rel.median())


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all equivariance tests passed")

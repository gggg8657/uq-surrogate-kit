"""Input-side summary features, used by the OOD detector and the shift probe.

Deliberately cheap and simulator-agnostic: anything shaped (n, C, H, W) gets a
fixed-length vector. The radial spectral bands are the part that matters --
the shifts a surrogate actually fails on are changes in how much energy sits at
high wavenumber, and that is invisible to mean/std alone.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def spectral_features(a, n_bands=8, n_ref=64):
    """(n, C, H, W) -> (n, C*(n_bands+3)) log band energies + moments.

    Band edges are fixed in *physical* wavenumber, spanning [0, sqrt(2)*n_ref/2]
    -- the Nyquist disc of the resolution the surrogate was trained at -- with
    one overflow band above it. That choice is what makes a 128^2 field
    comparable to a 64^2 one: the extra octave a finer grid resolves lands in
    the overflow band instead of rescaling every other band, so a detector
    fitted at 64^2 can score a resolution it has never seen without the
    features silently changing meaning underneath it.

    Energies are normalized by the total, so a pure amplitude change shows up in
    the moment features rather than smearing across the spectrum.
    """
    a = a.float()
    n, C, H, W = a.shape
    A = torch.fft.rfft2(a, norm="ortho")
    p = A.real**2 + A.imag**2
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=a.device).abs()
    kx = torch.fft.rfftfreq(W, d=1.0 / W, device=a.device)
    kr = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_max_ref = (2 ** 0.5) * n_ref / 2
    edges = torch.linspace(0, k_max_ref, n_bands + 1, device=a.device)
    tot = p.flatten(2).sum(-1).clamp_min(1e-20)
    feats = []
    for i in range(n_bands):
        m = (kr >= edges[i]) & (kr < edges[i + 1])
        feats.append(torch.log10((p * m).flatten(2).sum(-1) / tot + 1e-12))
    feats.append(torch.log10((p * (kr >= k_max_ref)).flatten(2).sum(-1) / tot + 1e-12))
    feats.append(torch.log10(a.flatten(2).std(-1, unbiased=False) + 1e-12))
    feats.append(a.flatten(2).abs().max(-1).values)
    return torch.stack(feats, dim=-1).reshape(n, -1)


class Mahalanobis:
    """Gaussian fit on in-distribution features; score is the squared distance.

    Fitted on *training* inputs only. It never sees an OOD set, so there is
    nothing for it to overfit to -- which is the point: the KPI requires a
    detector that was not tuned on what it is scored against.
    """

    def __init__(self, ridge=1e-3):
        self.ridge = ridge
        self.mu = None
        self.prec = None

    def fit(self, feat):
        f = feat.double()
        self.mu = f.mean(0)
        d = f - self.mu
        cov = d.T @ d / max(len(f) - 1, 1)
        cov = cov + self.ridge * torch.eye(cov.shape[0], dtype=cov.dtype,
                                           device=cov.device) * cov.diagonal().mean()
        self.prec = torch.linalg.inv(cov)
        return self

    def score(self, feat):
        d = feat.double() - self.mu
        return ((d @ self.prec) * d).sum(-1).float()

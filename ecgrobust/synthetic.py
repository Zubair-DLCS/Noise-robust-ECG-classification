"""Synthetic stand-in for MIT-BIH, used only for a fast smoke test (--synthetic).

Beats are sums of Gaussian P, QRS and T waves whose shapes differ by class;
noise mimics the three NSTDB types. Numbers from this mode mean nothing --
it only checks that the pipeline runs end to end.
"""
from __future__ import annotations

import numpy as np

from .data import PRE, POST, FS, BeatSet

_T = (np.arange(PRE + POST) - PRE) / FS  # seconds relative to R peak


def _g(c, w, a):
    return a * np.exp(-0.5 * ((_T - c) / w) ** 2)


def _beat(cls: int, rng) -> np.ndarray:
    j = lambda s: rng.normal(1, s)
    if cls == 0:    # normal
        b = _g(-0.18 * j(.05), .025, .15) + _g(0, .012 * j(.1), 1.0 * j(.1)) + _g(.25 * j(.05), .045, .3)
    elif cls == 1:  # supraventricular: early, abnormal P
        b = _g(-0.10 * j(.1), .02, -.08) + _g(0, .012 * j(.1), .95 * j(.1)) + _g(.22 * j(.05), .045, .28)
    elif cls == 2:  # ventricular: wide QRS, no P, inverted T
        b = _g(0, .035 * j(.1), 1.1 * j(.1)) + _g(.28 * j(.05), .06, -.35)
    else:           # fusion: in between
        b = _g(-0.17, .025, .08) + _g(0, .022 * j(.1), 1.0 * j(.1)) + _g(.26, .05, -.05)
    return b + rng.normal(0, .01, b.shape)


def make_beats(n_per_class=(3000, 400, 600, 150), n_records=20, seed=0) -> BeatSet:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c, n in enumerate(n_per_class):
        xs += [_beat(c, rng) for _ in range(n)]
        ys += [c] * n
    x, y = np.stack(xs).astype(np.float32), np.array(ys)
    rec = rng.integers(0, n_records, size=len(y))
    return BeatSet(x, y, rec)


def make_noise(seed=0, seconds=600) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    bw = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.11 * t + 1)
    ma = rng.normal(0, 1, t.size) * (1 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
    em = np.cumsum(rng.normal(0, 1, t.size)) * 0.01
    em += (rng.random(t.size) < 0.0005).cumsum() % 2 * rng.normal(0, 1)
    return {k: v.astype(np.float32) for k, v in {"bw": bw, "ma": ma, "em": em}.items()}

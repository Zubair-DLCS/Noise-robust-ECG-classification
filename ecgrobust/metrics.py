"""Accuracy, calibration and selective-prediction metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: weighted gap between confidence and accuracy per bin."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def reliability(probs, y, n_bins: int = 10):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    mids, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        mids.append((lo + hi) / 2)
        accs.append((pred[m] == y[m]).mean() if m.any() else np.nan)
        counts.append(int(m.sum()))
    return np.array(mids), np.array(accs), np.array(counts)


def risk_coverage(probs, y):
    """Selective prediction: keep the most confident beats first.

    Returns coverage, macro-F1 on kept beats, and the area under the
    error-rate/coverage curve (AURC, lower is better).
    """
    order = np.argsort(-probs.max(1))
    pred = probs.argmax(1)[order]
    yy = y[order]
    err = np.cumsum(pred != yy) / np.arange(1, len(yy) + 1)
    cov = np.arange(1, len(yy) + 1) / len(yy)
    grid = np.linspace(0.5, 1.0, 11)
    f1s = [macro_f1(yy[: max(1, int(c * len(yy)))], pred[: max(1, int(c * len(yy)))]) for c in grid]
    return grid, np.array(f1s), float(np.trapezoid(err, cov))


def summarise(logits, y, T: float = 1.0) -> dict:
    p = softmax(logits, T)
    pred = p.argmax(1)
    grid, f1s, aurc = risk_coverage(p, y)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": macro_f1(y, pred),
        "ece": ece(p, y),
        "mean_confidence": float(p.max(1).mean()),
        "aurc": aurc,
        "macro_f1_at_80pct_coverage": float(f1s[np.argmin(abs(grid - 0.8))]),
    }

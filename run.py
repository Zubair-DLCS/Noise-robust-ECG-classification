"""Noise-robustness and calibration audit of an ECG beat classifier.

Usage
  python run.py                    # full study on MIT-BIH (downloads ~110 MB on first run)
  python run.py --synthetic        # 1-minute smoke test on synthetic beats, no download

Trains two models on clean DS1 beats:
  clean   - trained on clean beats only
  noisy   - trained with real NSTDB noise injected at random SNRs (augmentation)
then evaluates both on DS2 (unseen patients) for every noise type x SNR level:
macro-F1, calibration (ECE, before/after temperature scaling), mean confidence
and selective prediction (abstain on low-confidence beats).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ecgrobust import data, metrics
from ecgrobust.model import BeatCNN, fit_temperature, predict_logits, train

SNRS = [None, 24, 18, 12, 6, 0, -6]   # None = clean


def split_noise(noise: dict) -> tuple[dict, dict]:
    """First half of each noise record for training augmentation, second half for testing.

    Test-time noise is therefore never seen during training."""
    tr = {k: v[: len(v) // 2] for k, v in noise.items()}
    te = {k: v[len(v) // 2:] for k, v in noise.items()}
    return tr, te


def make_augment(noise_tr: dict, p: float = 0.7, snr_range=(-6, 24)):
    keys = list(noise_tr)

    def aug(x_raw_batch, rng):
        x = x_raw_batch.copy()
        m = rng.random(len(x)) < p
        if m.any():
            k = keys[rng.integers(len(keys))]
            x[m] = data.add_noise(x[m], noise_tr[k], rng.uniform(*snr_range), rng)
        return data.normalise(x)
    return aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="physionet_data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rr", action="store_true", help="morphology only (version 1 of the study)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # ---------------- data ----------------
    if a.synthetic:
        from ecgrobust import synthetic
        allb = synthetic.make_beats(seed=a.seed)
        te_m = allb.rec >= 14
        va_m = (allb.rec >= 11) & ~te_m
        tr_m = ~te_m & ~va_m
        sub = lambda m: data.BeatSet(allb.x[m], allb.y[m], allb.rec[m], allb.rr[m])
        tr, va, te = sub(tr_m), sub(va_m), sub(te_m)
        noise = synthetic.make_noise(seed=a.seed)
        a.epochs = min(a.epochs, 4)
    else:
        data.download(a.data)
        ds1 = data.load_beats(a.data, data.DS1)
        is_va = np.isin(ds1.rec, data.VAL)
        tr = data.BeatSet(ds1.x[~is_va], ds1.y[~is_va], ds1.rec[~is_va], ds1.rr[~is_va])
        va = data.BeatSet(ds1.x[is_va], ds1.y[is_va], ds1.rec[is_va], ds1.rr[is_va])
        te = data.load_beats(a.data, data.DS2)
        noise = data.load_noise(a.data)
    noise_tr, noise_te = split_noise(noise)
    for name, s in [("train", tr), ("val", va), ("test", te)]:
        print(f"{name:5s}: {len(s.y):6d} beats  per class {np.bincount(s.y, minlength=4).tolist()}")

    x_va = data.normalise(va.x)

    # RR features: standardised with training statistics; empty when --no-rr.
    # They come from the reference R-peak annotations, so injected noise changes the
    # waveform but not the timing (R-peak detection under noise is out of scope).
    if a.no_rr:
        rr_tr, rr_va, rr_te = (np.zeros((len(s.y), 0), np.float32) for s in (tr, va, te))
    else:
        mu, sd = tr.rr.mean(0), tr.rr.std(0) + 1e-6
        rr_tr, rr_va, rr_te = (((s.rr - mu) / sd).astype(np.float32) for s in (tr, va, te))
    n_rr = rr_tr.shape[1]
    print(f"RR features: {'none (morphology only)' if a.no_rr else ', '.join(data.RR_NAMES)}")

    # ---------------- train ----------------
    models, temps = {}, {}
    for tag, aug in [("clean", None), ("noisy", make_augment(noise_tr))]:
        print(f"\nTraining '{tag}' model")
        x_tr = data.normalise(tr.x) if aug is None else tr.x   # augment fn normalises itself
        m = train(BeatCNN(n_rr=n_rr), x_tr, rr_tr, tr.y, x_va, rr_va, va.y,
                  epochs=a.epochs, seed=a.seed, augment_fn=aug)
        models[tag] = m
        temps[tag] = fit_temperature(predict_logits(m, x_va, rr_va), va.y)
        print(f"  temperature T = {temps[tag]:.2f}")

    # ---------------- evaluate ----------------
    rows = []
    store = {}   # raw logits kept for figures
    for nt in data.NOISE_TYPES:
        for snr in SNRS:
            if snr is None and nt != "bw":
                continue   # clean condition evaluated once
            rng = np.random.default_rng(1000 + a.seed)   # same noise for both models
            x = te.x if snr is None else data.add_noise(te.x, noise_te[nt], snr, rng)
            x = data.normalise(x)
            for tag, m in models.items():
                z = predict_logits(m, x, rr_te)
                store[(tag, "clean" if snr is None else nt, snr)] = z
                for cal, T in [("raw", 1.0), ("temp", temps[tag])]:
                    r = metrics.summarise(z, te.y, T)
                    r.update(model=tag, noise="clean" if snr is None else nt,
                             snr_db=snr, calibration=cal)
                    rows.append(r)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"temperatures": temps, "rows": rows, "synthetic": a.synthetic,
                   "rr_features": not a.no_rr, "test_class_counts": np.bincount(te.y, minlength=4).tolist()},
                  f, indent=1)

    from figures import make_figures
    make_figures(rows, store, te.y, temps, a.out)
    print_summary(rows)
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {a.out}/")


def print_summary(rows):
    get = lambda **k: next(r for r in rows if all(r[a] == b for a, b in k.items()))
    c = get(model="clean", noise="clean", snr_db=None, calibration="raw")
    print("\nPer-class F1, clean test, clean-trained: " + "  ".join(f"{k} {c['f1_' + k]:.3f}" for k in "NSVF"))
    print("\nMacro-F1 on unseen patients (raw softmax)")
    print(f"{'condition':22s} {'clean-trained':>14s} {'noise-trained':>14s}")
    conds = [("clean", None)] + [(nt, s) for nt in data.NOISE_TYPES for s in (6, 0)]
    for nt, s in conds:
        c = get(model="clean", noise=nt, snr_db=s, calibration="raw")["macro_f1"]
        n = get(model="noisy", noise=nt, snr_db=s, calibration="raw")["macro_f1"]
        print(f"{nt + ('' if s is None else f' @ {s} dB'):22s} {c:14.3f} {n:14.3f}")


if __name__ == "__main__":
    main()

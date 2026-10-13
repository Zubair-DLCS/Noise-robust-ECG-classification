"""Data loading for MIT-BIH Arrhythmia + MIT-BIH Noise Stress Test databases.

Beats are grouped into the four AAMI EC57 classes (N, S, V, F) and split with the
standard *inter-patient* scheme of de Chazal et al. (2004): train records (DS1)
and test records (DS2) never share a patient, so reported numbers are not
inflated by patient leakage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

FS = 360                     # sampling rate of both databases (Hz)
PRE, POST = 128, 128         # samples kept before / after each R peak (~0.36 s each)
CLASSES = ["N", "S", "V", "F"]

# AAMI EC57 mapping of MIT-BIH beat symbols (class Q / paced beats excluded).
AAMI = {
    **{s: 0 for s in ["N", "L", "R", "e", "j"]},
    **{s: 1 for s in ["A", "a", "J", "S"]},
    **{s: 2 for s in ["V", "E"]},
    "F": 3,
}

# de Chazal et al. inter-patient split (paced records 102, 104, 107, 217 excluded).
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
       205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234]
# Records of DS1 held out for early stopping and temperature scaling.
VAL = [108, 115, 205, 223]

NOISE_TYPES = {"bw": "baseline wander", "ma": "muscle artefact (EMG)", "em": "electrode motion"}


@dataclass
class BeatSet:
    x: np.ndarray        # (n, PRE+POST) raw millivolt beats, float32
    y: np.ndarray        # (n,) class index
    rec: np.ndarray      # (n,) record id


def download(root: str) -> None:
    """Fetch both databases from PhysioNet (~110 MB) unless already present."""
    import wfdb
    for db in ["mitdb", "nstdb"]:
        path = os.path.join(root, db)
        if not os.path.exists(os.path.join(path, "100.hea" if db == "mitdb" else "bw.hea")):
            print(f"Downloading {db} to {path} ...")
            wfdb.dl_database(db, path)


def load_beats(root: str, records: list[int]) -> BeatSet:
    import wfdb
    xs, ys, rs = [], [], []
    for r in records:
        base = os.path.join(root, "mitdb", str(r))
        sig = wfdb.rdrecord(base, channels=[0]).p_signal[:, 0].astype(np.float32)
        ann = wfdb.rdann(base, "atr")
        for pos, sym in zip(ann.sample, ann.symbol):
            if sym in AAMI and PRE <= pos < len(sig) - POST:
                xs.append(sig[pos - PRE: pos + POST])
                ys.append(AAMI[sym])
                rs.append(r)
    return BeatSet(np.stack(xs), np.array(ys), np.array(rs))


def load_noise(root: str) -> dict[str, np.ndarray]:
    """Real recorded noise (channel 0) from the Noise Stress Test Database."""
    import wfdb
    return {k: wfdb.rdrecord(os.path.join(root, "nstdb", k), channels=[0]).p_signal[:, 0].astype(np.float32)
            for k in NOISE_TYPES}


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add a random segment of `noise` to every beat at the requested SNR (dB).

    SNR is computed per beat: P_signal / P_noise, both after removing the mean.
    """
    n, L = x.shape
    starts = rng.integers(0, len(noise) - L, size=n)
    seg = np.stack([noise[s: s + L] for s in starts])
    seg = seg - seg.mean(1, keepdims=True)
    ps = ((x - x.mean(1, keepdims=True)) ** 2).mean(1, keepdims=True)
    pn = (seg ** 2).mean(1, keepdims=True) + 1e-12
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (x + scale * seg).astype(np.float32)


def normalise(x: np.ndarray) -> np.ndarray:
    """Per-beat z-score, applied *after* noise so the model sees what a device would."""
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)).astype(np.float32)

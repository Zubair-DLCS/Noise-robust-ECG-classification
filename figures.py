"""Figures for the README. Colours: validated categorical slots 1-2 (blue, orange)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecgrobust import metrics
from ecgrobust.data import NOISE_TYPES

COL = {"clean": "#2a78d6", "noisy": "#eb6834"}
MK = {"clean": "o", "noisy": "s"}
LBL = {"clean": "trained on clean beats", "noisy": "trained with noise augmentation"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
CLEAN_X = 30   # where the "clean" condition sits on the SNR axis

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _series(rows, model, noise, key, cal="raw"):
    clean = next(r for r in rows if r["model"] == model and r["noise"] == "clean" and r["calibration"] == cal)
    pts = sorted(((r["snr_db"], r[key]) for r in rows if r["model"] == model
                  and r["noise"] == noise and r["calibration"] == cal), reverse=True)
    return [CLEAN_X] + [p[0] for p in pts], [clean[key]] + [p[1] for p in pts]


def _snr_axis(ax):
    ticks = [CLEAN_X, 24, 18, 12, 6, 0, -6]
    ax.set_xticks(ticks, ["clean", "24", "18", "12", "6", "0", "−6"])
    ax.set_xlim(CLEAN_X + 2, -8)   # left = clean, right = heavier noise
    ax.set_xlabel("SNR (dB)  →  more noise")


def robustness(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row")
    for j, (nt, title) in enumerate(NOISE_TYPES.items()):
        for i, (key, ylabel) in enumerate([("macro_f1", "Macro-F1"), ("ece", "Calibration error (ECE)")]):
            ax = axes[i, j]
            for m in ("clean", "noisy"):
                x, y = _series(rows, m, nt, key)
                ax.plot(x, y, color=COL[m], marker=MK[m], markersize=6, label=LBL[m])
            _snr_axis(ax)
            if i == 0:
                ax.set_title(title, color=INK, fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Beat classification on unseen patients as real ECG noise increases", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_robustness.png"))
    plt.close(fig)


def reliability(store, y, temps, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (cond, snr, title) in zip(axes, [("clean", None, "Clean test beats"),
                                             ("ma", 0, "Muscle artefact at 0 dB")]):
        z = store[("clean", cond, snr)]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="perfect calibration")
        for T, style, col, mk, lab in [
                (1.0, "-", COL["clean"], "o", "raw softmax"),
                (temps["clean"], (0, (4, 2)), "#86b6ef", "s", f"temperature-scaled (T={temps['clean']:.2f})")]:
            mid, acc, cnt = metrics.reliability(metrics.softmax(z, T), y)
            ok = cnt >= 20   # hide bins too sparse to estimate accuracy
            ax.plot(mid[ok], acc[ok], ls=style, color=col, marker=mk, markersize=6, label=lab)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("Model confidence")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Observed accuracy")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Is the clean-trained model's confidence trustworthy?", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_reliability.png"))
    plt.close(fig)


def selective(store, y, out, cond=("em", 6)):
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for m in ("clean", "noisy"):
        grid, f1s, aurc = metrics.risk_coverage(metrics.softmax(store[(m, *cond)]), y)
        ax.plot(grid * 100, f1s, color=COL[m], marker=MK[m], markersize=6, label=f"{LBL[m]} (AURC {aurc:.3f})")
    ax.invert_xaxis()
    ax.set_xlabel("Coverage: % of beats the model chooses to classify")
    ax.set_ylabel("Macro-F1 on classified beats")
    ax.set_title(f"Abstaining on uncertain beats ({NOISE_TYPES[cond[0]]}, {cond[1]} dB)", color=INK, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_selective.png"))
    plt.close(fig)


CLS_COL = {"N": "#1baf7a", "S": "#eda100", "V": "#e87ba4"}   # categorical slots 3-5 (1-2 = models)
CLS_MK = {"N": "^", "S": "D", "V": "v"}
CLS_LBL = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular"}


def per_class(rows, out, noise="ma"):
    """Accuracy versus per-class F1 under one noise type: where an average hides a failure."""
    if "f1_N" not in rows[0]:
        return   # metrics from before per-class F1 was recorded
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, m in zip(axes, ("clean", "noisy")):
        labels = []
        x, acc = _series(rows, m, noise, "accuracy")
        ax.plot(x, acc, color=INK, ls=(0, (4, 2)), lw=1.6, marker="o", markersize=5)
        labels.append([acc[-1], "accuracy"])
        for c in "NSV":
            x, f = _series(rows, m, noise, f"f1_{c}")
            ax.plot(x, f, color=CLS_COL[c], marker=CLS_MK[c], markersize=7)
            labels.append([f[-1], f"{CLS_LBL[c]} F1"])
        labels.sort()
        for i in range(1, len(labels)):          # keep end labels at least 0.06 apart
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.06)
        for yv, text in labels:
            ax.text(x[-1] - 1.2, yv, text, va="center", color=INK, fontsize=9)
        _snr_axis(ax)
        ax.set_xlim(CLEAN_X + 2, -18)   # room for direct labels on the right
        ax.set_xticks([CLEAN_X, 24, 18, 12, 6, 0, -6], ["clean", "24", "18", "12", "6", "0", "−6"])
        ax.set_title(LBL[m].capitalize(), color=INK, fontsize=11)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Score on unseen patients")
    fig.suptitle(f"Under {NOISE_TYPES[noise]}, accuracy stays high while ventricular-beat detection collapses",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_per_class.png"))
    plt.close(fig)


def make_figures(rows, store, y, temps, out):
    per_class(rows, out)
    robustness(rows, out)
    reliability(store, y, temps, out)
    selective(store, y, out)

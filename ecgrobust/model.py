"""Beat classifier (1D CNN on morphology + small MLP on RR timing), training, temperature scaling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeatCNN(nn.Module):
    """1D CNN over the beat waveform, fused with an MLP over RR-interval features.

    n_rr = 0 gives the morphology-only model (version 1 of this study).
    """

    def __init__(self, n_classes: int = 4, n_rr: int = 6, width: int = 32):
        super().__init__()
        self.n_classes, self.n_rr = n_classes, n_rr

        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(1, width, 9), block(width, width, 7),
                                      block(width, 2 * width, 5), block(2 * width, 2 * width, 3),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten())
        rr_dim = 16 if n_rr else 0
        self.rr = nn.Sequential(nn.Linear(n_rr, rr_dim), nn.ReLU()) if n_rr else None
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * width + rr_dim, n_classes))

    def forward(self, x, rr):                    # x: (B, L), rr: (B, n_rr)
        h = self.features(x.unsqueeze(1))
        if self.rr is not None:
            h = torch.cat([h, self.rr(rr)], 1)
        return self.head(h)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(np.sqrt(w), dtype=torch.float32)   # sqrt: softens extreme imbalance


@torch.no_grad()
def predict_logits(model, x: np.ndarray, rr: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    return torch.cat([model(torch.from_numpy(x[i:i + batch]), torch.from_numpy(rr[i:i + batch]))
                      for i in range(0, len(x), batch)]).numpy()


def train(model, x_tr, rr_tr, y_tr, x_va, rr_va, y_va, epochs=15, lr=2e-3, batch=256, seed=0,
          augment_fn=None, log=print):
    """Weighted cross-entropy + AdamW; keeps the epoch with best validation macro-F1.

    augment_fn(x_batch_numpy, rng) -> x_batch_numpy is applied to the waveform only.
    """
    from .metrics import macro_f1
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = class_weights(y_tr, model.n_classes)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            xb = x_tr[b] if augment_fn is None else augment_fn(x_tr[b], rng)
            out = model(torch.from_numpy(xb), torch.from_numpy(rr_tr[b]))
            loss = F.cross_entropy(out, torch.from_numpy(y_tr[b]), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        f1 = macro_f1(y_va, predict_logits(model, x_va, rr_va).argmax(1))
        log(f"  epoch {ep + 1:2d}/{epochs}  val macro-F1 {f1:.3f}")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Guo et al. (2017): one scalar T minimising validation NLL."""
    z, t = torch.from_numpy(logits), torch.from_numpy(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))

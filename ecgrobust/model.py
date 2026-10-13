"""1D CNN beat classifier, training loop and temperature scaling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeatCNN(nn.Module):
    """Small residual-free 1D CNN (~60k parameters); fast on a laptop CPU."""

    def __init__(self, n_classes: int = 4, width: int = 32):
        super().__init__()
        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(1, width, 9), block(width, width, 7),
                                      block(width, 2 * width, 5), block(2 * width, 2 * width, 3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(2 * width, n_classes))

    def forward(self, x):                       # x: (B, L)
        return self.head(self.features(x.unsqueeze(1)))


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(np.sqrt(w), dtype=torch.float32)   # sqrt: softens extreme imbalance


@torch.no_grad()
def predict_logits(model, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    return torch.cat([model(torch.from_numpy(x[i:i + batch])) for i in range(0, len(x), batch)]).numpy()


def train(model, x_tr, y_tr, x_va, y_va, epochs=15, lr=2e-3, batch=256, seed=0,
          augment_fn=None, log=print):
    """Weighted cross-entropy + AdamW; keeps the epoch with best validation macro-F1.

    augment_fn(x_batch_numpy, rng) -> x_batch_numpy, applied per batch when given.
    """
    from .metrics import macro_f1
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = class_weights(y_tr, model.head[-1].out_features)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            xb = x_tr[b] if augment_fn is None else augment_fn(x_tr[b], rng)
            loss = F.cross_entropy(model(torch.from_numpy(xb)), torch.from_numpy(y_tr[b]), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        f1 = macro_f1(y_va, predict_logits(model, x_va).argmax(1))
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

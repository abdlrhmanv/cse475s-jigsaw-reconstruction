"""Ranking metrics for imbalanced neighbour classification."""

from __future__ import annotations

import numpy as np


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (higher is better)."""
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = float(y.sum())
    if n_pos <= 0 or len(y) == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def binary_pr_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """F1 at 0.5, F1 at the best threshold, AP, and mean scores by class."""
    p = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int32).ravel()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    p_pos = float(p[y == 1].mean()) if n_pos else 0.0
    p_neg = float(p[y == 0].mean()) if n_neg else 0.0

    def at_threshold(t: float) -> tuple[float, float, float, int, int, int]:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        return f1, prec, rec, tp, fp, fn

    f1_05, prec_05, rec_05, tp, fp, fn = at_threshold(0.5)
    best_f1, best_t, best_prec, best_rec = f1_05, 0.5, prec_05, rec_05
    for t in np.linspace(0.05, 0.95, 19):
        f1, prec, rec, _, _, _ = at_threshold(float(t))
        if f1 > best_f1:
            best_f1, best_t, best_prec, best_rec = f1, float(t), prec, rec
    return {
        "f1": f1_05,
        "prec": prec_05,
        "rec": rec_05,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "f1_best": best_f1,
        "t_best": best_t,
        "ap": average_precision(p, y),
        "p_pos": p_pos,
        "p_neg": p_neg,
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
    }

"""Three-way deferral decisions and metrics."""
from __future__ import annotations

import numpy as np


def deferral_labels(prob: np.ndarray, t_low: float, t_high: float, present_label: str = "present") -> np.ndarray:
    absent_label = "absent" if present_label == "present" else "normal"
    labels = np.full(len(prob), "defer", dtype=object)
    labels[prob < t_low] = absent_label
    labels[prob > t_high] = present_label
    return labels


def deferral_metrics(prob: np.ndarray, y_true: np.ndarray, t_low: float, t_high: float) -> dict:
    """Metrics among automatically classified recordings (non-deferred)."""
    labels = deferral_labels(prob, t_low, t_high)
    auto = labels != "defer"
    n_auto = int(auto.sum())
    if n_auto == 0:
        return {"coverage": 0.0, "n_defer": len(prob), "fn_auto": 0, "fp_auto": 0}

    pred_present = labels[auto] == "present"
    true_present = y_true[auto] == 1
    tp = int((pred_present & true_present).sum())
    fn = int((~pred_present & true_present).sum())
    tn = int((~pred_present & ~true_present).sum())
    fp = int((pred_present & ~true_present).sum())

    return {
        "coverage": n_auto / len(prob),
        "n_defer": int((labels == "defer").sum()),
        "fn_auto": fn,
        "fp_auto": fp,
        "sensitivity_auto": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity_auto": tn / (tn + fp) if (tn + fp) else float("nan"),
    }


def forced_fn_count(prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> int:
    return int(((y_true == 1) & (prob < threshold)).sum())

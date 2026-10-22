import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def scores(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion": confusion_matrix(y, pred).tolist(),
        "n": int(len(y)),
    }


def learning_curve_points(X, y, sizes, fit_fn, proba_fn):
    xs, train_f1, val_f1 = [], [], []
    n = len(y)
    cut = int(n * 0.8)
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]
    for frac in sizes:
        k = max(50, int(len(y_tr) * frac))
        model = fit_fn(X_tr[:k], y_tr[:k])
        p_tr = proba_fn(model, X_tr[:k])
        p_va = proba_fn(model, X_va)
        train_f1.append(float(f1_score(y_tr[:k], (p_tr >= 0.5).astype(int), zero_division=0)))
        val_f1.append(float(f1_score(y_va, (p_va >= 0.5).astype(int), zero_division=0)))
        xs.append(k)
    return xs, train_f1, val_f1


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def roc_points(y, proba):
    fpr, tpr, _ = roc_curve(y, proba)
    return fpr.tolist(), tpr.tolist()

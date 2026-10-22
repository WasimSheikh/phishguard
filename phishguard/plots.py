from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def roc_chart(curves, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, y, proba in curves:
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on the held-out test set")
    ax.legend(loc="lower right")
    return _save(fig, path)


def confusion_chart(y, pred, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False)
    ax.set_title(title)
    return _save(fig, path)


def loss_chart(history, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(np.arange(1, len(history) + 1), history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss (cross-entropy + L2)")
    ax.set_title("NumPy gradient descent")
    return _save(fig, path)


def learning_curve_chart(sizes, train_f1, val_f1, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(sizes, train_f1, label="train F1")
    ax.plot(sizes, val_f1, label="held-out F1")
    ax.set_xlabel("Training rows")
    ax.set_ylabel("F1")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Learning curve (NumPy model)")
    ax.legend()
    return _save(fig, path)

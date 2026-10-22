import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def binary_cross_entropy(y, p, w, l2):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    ce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return ce + (l2 / (2.0 * len(y))) * np.sum(w * w)


def fit(X, y, lr=0.5, epochs=400, l2=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []
    y = y.astype(float)
    for _ in range(epochs):
        p = predict_proba(X, w, b)
        err = p - y
        w -= lr * ((X.T @ err) / n + (l2 / n) * w)
        b -= lr * np.mean(err)
        history.append(binary_cross_entropy(y, p, w, l2))
    return w, b, np.array(history)


def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)


def best_threshold(y, proba, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    y = y.astype(int)
    for t in grid:
        pred = (proba >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1

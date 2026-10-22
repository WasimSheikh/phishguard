import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from phishguard.data import load_urls, split_xy
from phishguard.logistic import best_threshold, fit, predict_proba
from phishguard.metrics import dump, learning_curve_points, scores
from phishguard.plots import confusion_chart, learning_curve_chart, loss_chart, roc_chart

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "phishguard.npz"
METRICS = ROOT / "reports" / "metrics.json"
SAMPLE = 80000


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def run(sample=SAMPLE, seed=7):
    df = load_urls(sample=sample, seed=seed)
    X_train, y_train, X_val, y_val, X_test, y_test, names = split_xy(df, seed=seed)
    X_train_s, mean, std = standardize(X_train)
    X_val_s, _, _ = standardize(X_val, mean, std)
    X_test_s, _, _ = standardize(X_test, mean, std)

    w, b, history = fit(X_train_s, y_train, lr=0.8, epochs=350, l2=0.05)
    np_val = predict_proba(X_val_s, w, b)
    np_thr, _ = best_threshold(y_val, np_val)
    np_test = predict_proba(X_test_s, w, b)

    sk = LogisticRegression(max_iter=500)
    sk.fit(X_train_s, y_train)
    sk_val = sk.predict_proba(X_val_s)[:, 1]
    sk_thr, _ = best_threshold(y_val, sk_val)
    sk_test = sk.predict_proba(X_test_s)[:, 1]

    results = {
        "dataset": "PhiUSIIL Phishing URL Dataset (UCI / Mendeley)",
        "license": "CC BY 4.0",
        "sample": int(len(df)),
        "split": "70/15/15 stratified",
        "numpy": scores(y_test, np_test, np_thr),
        "sklearn": scores(y_test, sk_test, sk_thr),
        "features": names,
    }

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL,
        w=w,
        b=np.array([b]),
        mean=mean,
        std=std,
        threshold=np.array([np_thr]),
        names=np.array(names),
    )

    charts = ROOT / "charts"
    roc_chart(
        [("NumPy LR", y_test, np_test), ("sklearn LR", y_test, sk_test)],
        charts / "roc.png",
    )
    confusion_chart(y_test, (np_test >= np_thr).astype(int), "NumPy LR", charts / "confusion_numpy.png")
    confusion_chart(y_test, (sk_test >= sk_thr).astype(int), "sklearn LR", charts / "confusion_sklearn.png")
    loss_chart(history, charts / "loss.png")

    def _fit(Xa, ya):
        ww, bb, _ = fit(Xa, ya, lr=0.8, epochs=200, l2=0.05)
        return ww, bb

    def _proba(model, Xa):
        ww, bb = model
        return predict_proba(Xa, ww, bb)

    # keep the curve cheap: shuffle copy, then growing prefixes
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y_train))
    cap = min(12000, len(idx))
    xs, tr_f1, va_f1 = learning_curve_points(
        X_train_s[idx[:cap]], y_train[idx[:cap]], [0.1, 0.25, 0.4, 0.6, 0.8, 1.0], _fit, _proba
    )
    learning_curve_chart(xs, tr_f1, va_f1, charts / "learning_curve.png")

    dump(METRICS, results)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()

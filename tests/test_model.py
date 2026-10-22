import numpy as np

from phishguard.logistic import best_threshold, binary_cross_entropy, fit, predict, predict_proba, sigmoid


def test_sigmoid_bounds():
    s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert s[0] < 1e-6
    assert abs(s[1] - 0.5) < 1e-9
    assert s[2] > 1 - 1e-6


def test_fit_separates_two_blobs():
    rng = np.random.RandomState(0)
    X0 = rng.normal([-2, -2], 0.4, size=(80, 2))
    X1 = rng.normal([2, 2], 0.4, size=(80, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 80 + [1] * 80)
    w, b, hist = fit(X, y, lr=0.8, epochs=250, l2=0.01)
    acc = np.mean(predict(X, w, b) == y)
    assert acc > 0.95
    assert hist[-1] < hist[0]
    p = predict_proba(X, w, b)
    assert binary_cross_entropy(y, p, w, 0.01) < 0.2


def test_threshold_prefers_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.55, 0.58, 0.9, 0.95])
    t, f1 = best_threshold(y, proba, grid=np.array([0.3, 0.56, 0.8]))
    assert t == 0.56
    assert f1 > 0.7

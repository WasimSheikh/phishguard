import numpy as np

from phishguard.__main__ import score_url
from phishguard.features import NAMES
from phishguard.logistic import fit


def test_score_url_with_tiny_model(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, len(NAMES)))
    y = (X[:, 0] + X[:, -1] > 0).astype(int)
    w, b, _ = fit(X, y, lr=0.5, epochs=80, l2=0.01)
    path = tmp_path / "m.npz"
    np.savez(
        path,
        w=w,
        b=np.array([b]),
        mean=np.zeros(len(NAMES)),
        std=np.ones(len(NAMES)),
        threshold=np.array([0.5]),
        names=np.array(NAMES),
    )
    bundle = np.load(path, allow_pickle=True)
    proba, thr, top, feats = score_url("https://secure-login.example.com/verify", bundle)
    assert 0.0 <= proba <= 1.0
    assert thr == 0.5
    assert "https" not in feats
    assert feats["n_suspicious"] >= 1
    assert len(top) == 5

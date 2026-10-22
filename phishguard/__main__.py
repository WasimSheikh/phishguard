import argparse
import sys
from pathlib import Path

import numpy as np

from phishguard.features import extract, vector
from phishguard.logistic import predict_proba

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "phishguard.npz"


def load_model(path=MODEL):
    if not Path(path).exists():
        sys.exit("no trained model yet; run: python -m phishguard train")
    data = np.load(path, allow_pickle=True)
    return data


def score_url(url, bundle=None):
    bundle = bundle if bundle is not None else load_model()
    names = [str(n) for n in bundle["names"]]
    raw = np.asarray(vector(url), dtype=float)
    x = (raw - bundle["mean"]) / bundle["std"]
    w = bundle["w"]
    b = float(bundle["b"][0]) if bundle["b"].shape else float(bundle["b"])
    proba = float(predict_proba(x.reshape(1, -1), w, b)[0])
    contrib = w * x
    order = np.argsort(-np.abs(contrib))
    top = [(names[i], float(contrib[i]), float(raw[i])) for i in order[:5]]
    threshold = float(bundle["threshold"][0])
    return proba, threshold, top, extract(url)


def print_score(url):
    proba, threshold, top, _ = score_url(url)
    label = "phishing" if proba >= threshold else "likely legit"
    print(f"{url}")
    print(f"probability {proba:.3f}  threshold {threshold:.2f}  -> {label}")
    print("top features (weight * standardized value):")
    for name, c, raw in top:
        print(f"  {name:14}  {c:+.3f}   raw={raw:g}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="phishguard")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score one URL")
    s.add_argument("url")
    t = sub.add_parser("train", help="download data and fit both models")
    t.add_argument("--sample", type=int, default=80000)
    args = p.parse_args(argv)
    if args.cmd == "train":
        from phishguard.train import run

        run(sample=args.sample)
        return
    print_score(args.url)


if __name__ == "__main__":
    main()

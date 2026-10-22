"""Write the saved NumPy weights to demo/model.json for the static demo."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phishguard.features import NAMES, SUSPICIOUS  # noqa: E402

THRESHOLD = 0.35


def export(src=ROOT / "models" / "phishguard.npz", dst=ROOT / "demo" / "model.json"):
    data = np.load(src, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    assert tuple(names) == NAMES, names
    b = data["b"]
    model = {
        "features": names,
        "weights": [float(v) for v in data["w"]],
        "bias": float(b[0]) if b.shape else float(b),
        "mean": [float(v) for v in data["mean"]],
        "std": [float(v) for v in data["std"]],
        "threshold": THRESHOLD,
        "suspicious_words": list(SUSPICIOUS),
    }
    Path(dst).write_text(json.dumps(model, indent=2) + "\n")
    return model


if __name__ == "__main__":
    export()
    print("wrote demo/model.json")

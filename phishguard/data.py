import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from phishguard.features import NAMES, vector

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "urls.csv"

# Prasad & Chandra, Computers & Security 2024. UCI + Mendeley, CC BY 4.0
UCI_ZIP = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"


def download(dest=DATA_CSV):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    print(f"downloading {UCI_ZIP}")
    with urlopen(UCI_ZIP, timeout=180) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        raw = zf.read(member)

    df = pd.read_csv(io.BytesIO(raw), usecols=["URL", "label"])
    df = df.dropna(subset=["URL", "label"])
    # PhiUSIIL: 1 = legitimate, 0 = phishing
    df["phishing"] = (df["label"] == 0).astype(int)
    df[["URL", "phishing"]].to_csv(dest, index=False)
    print(f"wrote {len(df)} rows to {dest}")
    return dest


def load_urls(path=DATA_CSV, sample=None, seed=7):
    path = download(path)
    df = pd.read_csv(path)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def featurize(urls):
    rows = [vector(u) for u in urls]
    return np.asarray(rows, dtype=float), list(NAMES)


def split_xy(df, seed=7):
    X, names = featurize(df["URL"].tolist())
    y = df["phishing"].to_numpy(dtype=int)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest
    )
    return (X_train, y_train, X_val, y_val, X_test, y_test, names)

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from phishguard.__main__ import load_model, score_url
from phishguard.features import extract

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"

URLS = [
    "https://example.com",
    "example.com",
    "http://example.com/",
    "  https://example.com/path  ",
    "https://www.google.com/search?q=phishing+test",
    "https://github.com/WasimSheikh/phishguard",
    "http://192.168.0.14/secure-login/verify-account",
    "https://update-confirm.example.net/wallet/invoice",
    "paypal.com.account-verify.login.example.ru/signin?id=88213",
    "http://10.0.0.1:8080/admin",
    "http://[2001:db8::1]/login",
    "http://[::1]:8000/",
    "https://user:pass@bank.example.com/account",
    "https://a.b.c.d.e.example.co.uk/x",
    "ftp://files.example.org/pub/file-2024.tar.gz",
    "HTTPS://EXAMPLE.COM/LOGIN",
    "https://secure-banking-update.com/confirm?acct=1234567890",
    "https://bit.ly/3xYz9Q",
    "https://example.com/#fragment-only",
    "https://example.com?query=1&b=2",
    "https://xn--pple-43d.com/",
    "https://münchen.example.de/straße",
    "http://999.999.999.999/",
    "https://example.com:443/path;params?x=y",
    "www.amazon.com/gp/your-account/order-history",
    "https://login.microsoftonline.com.verify-session.info/",
    "https://docs.python.org/3/library/urllib.parse.html",
    "http://free-gift-card-2026.win/claim/now!!!",
    "https://example.com/" + "a" * 300,
    "https://192.168.1.1.nip.io/password-reset",
    "mailto:someone@example.com",
    "https://drive.google.com/file/d/1AbC_dEf-GhI/view?usp=sharing",
    "localhost:3000/suspend/account",
    "http://example..com//double//slash",
    "https://example.com/%70%61%79%70%61%6c",
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_scores_match_python():
    script = (
        "const pg = require(process.argv[1]);"
        "const model = require(process.argv[2]);"
        "const urls = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        "process.stdout.write(JSON.stringify(urls.map(u => {"
        "  const r = pg.score(u, model); return [r.probability, r.features]; })));"
    )
    proc = subprocess.run(
        ["node", "-e", script, str(DEMO / "phishguard.js"), str(DEMO / "model.json")],
        input=json.dumps(URLS),
        capture_output=True,
        text=True,
        check=True,
    )
    js = json.loads(proc.stdout)
    bundle = load_model()
    assert len(js) == len(URLS) >= 30
    for url, (p_js, feats_js) in zip(URLS, js):
        p_py = score_url(url, bundle)[0]
        assert feats_js == extract(url), url
        assert abs(p_js - p_py) < 1e-6, url


def test_model_json_matches_npz():
    model = json.loads((DEMO / "model.json").read_text())
    bundle = load_model()
    assert model["features"] == [str(n) for n in bundle["names"]]
    assert model["weights"] == pytest.approx(list(bundle["w"]))
    assert model["threshold"] == 0.35

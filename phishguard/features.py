import re
from urllib.parse import urlparse

SUSPICIOUS = (
    "login",
    "signin",
    "verify",
    "account",
    "secure",
    "update",
    "confirm",
    "password",
    "banking",
    "paypal",
    "wallet",
    "invoice",
    "suspend",
)

_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_IPV6 = re.compile(r"^\[?[0-9a-fA-F:]+\]?$")
_SPECIAL = re.compile(r"[^A-Za-z0-9]")

NAMES = (
    "url_length",
    "n_digits",
    "n_special",
    "n_subdomains",
    "ip_in_host",
    "https",
    "n_suspicious",
)


def _host(url):
    raw = url.strip()
    if "://" not in raw:
        raw = "http://" + raw
    return urlparse(raw)


def extract(url):
    parsed = _host(url)
    host = (parsed.hostname or "").lower()
    text = url.strip()
    parts = [p for p in host.split(".") if p]
    n_sub = max(len(parts) - 2, 0)
    ip = 1.0 if host and (_IPV4.match(host) or (":" in host and _IPV6.match(host))) else 0.0
    return {
        "url_length": float(len(text)),
        "n_digits": float(sum(c.isdigit() for c in text)),
        "n_special": float(len(_SPECIAL.findall(text))),
        "n_subdomains": float(n_sub),
        "ip_in_host": ip,
        "https": 1.0 if parsed.scheme == "https" else 0.0,
        "n_suspicious": float(sum(w in text.lower() for w in SUSPICIOUS)),
    }


def vector(url):
    feats = extract(url)
    return [feats[name] for name in NAMES]

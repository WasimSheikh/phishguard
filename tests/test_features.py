from phishguard.features import NAMES, extract, vector


def test_scheme_is_stripped():
    a = extract("https://example.com/login")
    b = extract("http://example.com/login")
    assert a == b
    assert "https" not in a
    assert a["url_length"] == len("example.com/login")
    assert a["n_suspicious"] >= 1


def test_ip_in_host():
    feats = extract("http://192.168.0.10/update")
    assert feats["ip_in_host"] == 1.0


def test_subdomains():
    feats = extract("https://a.b.evil.example.com")
    assert feats["n_subdomains"] == 3


def test_vector_order():
    v = vector("https://bank-secure.example.com/confirm")
    assert "https" not in NAMES
    assert len(v) == len(NAMES)
    assert v[NAMES.index("n_suspicious")] >= 2

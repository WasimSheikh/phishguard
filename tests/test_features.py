from phishguard.features import NAMES, extract, vector


def test_https_and_length():
    feats = extract("https://example.com/login")
    assert feats["https"] == 1.0
    assert feats["url_length"] == len("https://example.com/login")
    assert feats["n_suspicious"] >= 1


def test_ip_host_and_http():
    feats = extract("http://192.168.0.10/update")
    assert feats["ip_in_host"] == 1.0
    assert feats["https"] == 0.0


def test_subdomains():
    feats = extract("https://a.b.evil.example.com")
    assert feats["n_subdomains"] == 3


def test_vector_order():
    v = vector("https://bank-secure.example.com/confirm")
    assert len(v) == len(NAMES)
    assert v[NAMES.index("https")] == 1.0
    assert v[NAMES.index("n_suspicious")] >= 2

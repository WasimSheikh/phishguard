from app import score


def test_score_returns_probability():
    proba, label, top, threshold = score("https://example.com")
    assert 0.0 <= proba <= 1.0
    assert label in {"phishing", "likely legit"}
    assert 0.0 < threshold < 1.0
    assert len(top) == 5

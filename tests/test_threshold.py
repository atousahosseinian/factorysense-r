from factorysense.calibration.threshold_calibrator import (
    quantile_threshold,
    classify_score,
    risk_level,
)


def test_quantile_threshold_returns_float():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold = quantile_threshold(scores, quantile=0.8)

    assert isinstance(threshold, float)
    assert threshold > 0


def test_classify_score():
    assert classify_score(0.9, 0.5) == "Reject"
    assert classify_score(0.2, 0.5) == "Pass"


def test_risk_level():
    assert risk_level(0.9, 0.5) == "High"
    assert risk_level(0.4, 0.5) == "Medium"
    assert risk_level(0.1, 0.5) == "Low"

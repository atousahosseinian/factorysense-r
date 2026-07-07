import numpy as np


def quantile_threshold(normal_scores, quantile=0.95):
    """
    Compute a threshold from normal validation anomaly scores.

    Scores above this threshold are considered suspicious/anomalous.
    """
    scores = np.asarray(normal_scores, dtype=float)

    if scores.size == 0:
        raise ValueError("normal_scores cannot be empty")

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")

    return float(np.quantile(scores, quantile))


def classify_score(score, threshold):
    """
    Convert an anomaly score into a Pass/Reject decision.
    """
    return "Reject" if score >= threshold else "Pass"


def risk_level(score, threshold):
    """
    Assign a simple risk level based on the calibrated threshold.
    """
    if score >= threshold:
        return "High"
    if score >= threshold * 0.75:
        return "Medium"
    return "Low"

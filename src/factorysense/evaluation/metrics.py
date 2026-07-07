import numpy as np


def binary_classification_counts(y_true, y_pred):
    """
    Compute TP, TN, FP, FN for binary classification.

    Convention:
    1 = anomaly / defective
    0 = normal / good
    """
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def precision_recall_f1(y_true, y_pred):
    """
    Compute precision, recall, and F1-score.
    """
    counts = binary_classification_counts(y_true, y_pred)

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

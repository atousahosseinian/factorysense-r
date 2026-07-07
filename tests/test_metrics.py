from factorysense.evaluation.metrics import (
    binary_classification_counts,
    precision_recall_f1,
)


def test_binary_classification_counts():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]

    counts = binary_classification_counts(y_true, y_pred)

    assert counts["tn"] == 1
    assert counts["fp"] == 1
    assert counts["tp"] == 1
    assert counts["fn"] == 1


def test_precision_recall_f1():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]

    result = precision_recall_f1(y_true, y_pred)

    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5

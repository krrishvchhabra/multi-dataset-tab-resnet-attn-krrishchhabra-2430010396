import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_classification(y_true, y_pred, y_prob):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    try:
        if y_prob.ndim == 1:
            results["roc_auc"] = roc_auc_score(
                y_true, y_prob
            )
            results["average_precision"] = (
                average_precision_score(y_true, y_prob)
            )
        else:
            results["roc_auc_macro_ovr"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        results["roc_auc"] = np.nan

    return results

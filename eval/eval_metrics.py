from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def binarize_predictions(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_prob >= threshold).astype(int)

def compute_overall_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray]:
    y_pred = binarize_predictions(y_prob, threshold)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return metrics, y_pred

def compute_per_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_pred = binarize_predictions(y_prob, threshold)

    rows = []
    for idx, label in enumerate(label_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        rows.append(
            {
                "label": label,
                "f1": f1_score(yt, yp, zero_division=0),
                "support": int(np.sum(yt)),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["f1", "support"], ascending=[False, False]).reset_index(drop=True)
    return df

def compute_truth_table_counts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_pred = binarize_predictions(y_prob, threshold)

    rows = []
    for idx, label in enumerate(label_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        rows.append(
            {
                "label": label,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
            }
        )

    return pd.DataFrame(rows)
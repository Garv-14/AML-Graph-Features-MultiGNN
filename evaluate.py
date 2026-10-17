"""Shared evaluation module for Anti-Money Laundering (AML) models.

Provides standardized metric calculations, precision-recall curve threshold
optimization, alert budget metrics, and experiment logging across all model families
(tabular, graph-feature, and GNN).
"""

import os
import csv
from typing import Dict, Any, Tuple, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)


def evaluate_scores(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
    model_name: str = None,
    split_name: str = None,
) -> Dict[str, Any]:
    """Evaluates continuous prediction scores against binary ground-truth labels.

    Computes rank-based discrimination metrics (PR-AUC, ROC-AUC) as well as
    threshold-dependent classification metrics (Precision, Recall, F1, Confusion Matrix)
    at a specified operating point. Gracefully handles single-class edge cases.

    Args:
        y_true: Binary ground-truth labels (0 for legitimate, 1 for laundering).
        y_score: Predicted probabilities or continuous risk scores for the positive class (1).
        threshold: Decision threshold for converting continuous scores to binary predictions.
        model_name: Optional name/identifier of the model being evaluated.
        split_name: Optional name of the data split (e.g., 'val', 'test').

    Returns:
        Dictionary containing transaction counts, prevalence, discrimination metrics,
        and threshold-specific confusion matrix counts.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    n_rows = int(len(y_true_arr))
    n_positives = int(np.sum(y_true_arr == 1))
    prevalence = float(n_positives / n_rows) if n_rows > 0 else 0.0

    # Binary predictions at threshold
    y_pred = (y_score_arr >= threshold).astype(int)

    # Threshold-specific classification metrics (positive class = 1)
    prec = float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0))

    # Confusion matrix elements
    if len(np.unique(y_true_arr)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    else:
        # Single-class edge case
        if y_true_arr[0] == 1:
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))
            tn, fp = 0, 0
        else:
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            tp, fn = 0, 0

    # Rank-based metrics: PR-AUC (Average Precision) and ROC-AUC
    # Handle single-class edge case gracefully where AUC is undefined
    if n_positives == 0 or n_positives == n_rows:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        try:
            pr_auc = float(average_precision_score(y_true_arr, y_score_arr))
        except ValueError:
            pr_auc = float("nan")
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
        except ValueError:
            roc_auc = float("nan")

    result = {
        "model_name": model_name,
        "split_name": split_name,
        "n_rows": n_rows,
        "n_positives": n_positives,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    return result


def best_f1_threshold(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Finds the score threshold that maximizes the F1 score on the Precision-Recall curve.

    In heavily class-imbalanced settings like AML, the default 0.5 probability threshold
    is typically sub-optimal. This function sweeps the PR curve to find the optimal
    trade-off between precision and recall.

    Note:
        To prevent leakage, this threshold MUST be calculated strictly on the
        VALIDATION split, and then applied unchanged to the TEST split.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_score: Continuous predicted risk scores or probabilities.

    Returns:
        Optimal threshold as a float (falls back to 0.5 if no valid threshold found).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    if np.sum(y_true_arr == 1) == 0:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_score_arr)

    # precision_recall_curve returns precisions and recalls of length len(thresholds) + 1
    # Slice to align with thresholds
    p = precisions[:-1]
    r = recalls[:-1]

    # Compute F1 for each threshold avoiding division by zero
    denominator = p + r
    f1_scores = np.where(denominator > 0, (2.0 * p * r) / denominator, 0.0)

    if len(f1_scores) == 0:
        return 0.5

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def alert_budget_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    budgets: Sequence[float] = (0.001, 0.005, 0.01),
) -> Dict[str, Dict[str, Any]]:
    """Evaluates model performance under realistic operational investigation alert budgets.

    Compliance and AML investigation teams can only review a fixed percentage of transactions
    per day (e.g. top 0.1%, 0.5%, or 1.0% highest-risk alerts). This function computes
    Precision@k and Recall@k when flagging exactly the top-k highest-scoring transactions.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_score: Predicted risk scores.
        budgets: Sequence of alert budget fractions (e.g. 0.001 for top 0.1%).

    Returns:
        Dictionary mapping budget string (e.g. 'budget_0.001') to a sub-dictionary with:
        budget, k (number of alerts generated), precision_at_k, and recall_at_k.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    n_rows = len(y_true_arr)
    n_positives = int(np.sum(y_true_arr == 1))

    # Rank indices in descending order of predicted score
    sorted_indices = np.argsort(-y_score_arr)

    results = {}
    for b in budgets:
        # Number of alerts to inspect: at least 1 alert if budget > 0 and n_rows > 0
        k = max(1, int(round(n_rows * b))) if n_rows > 0 else 0
        k = min(k, n_rows)

        if k == 0:
            p_at_k = 0.0
            r_at_k = 0.0
        else:
            top_k_indices = sorted_indices[:k]
            positives_found = int(np.sum(y_true_arr[top_k_indices] == 1))
            p_at_k = float(positives_found / k)
            r_at_k = float(positives_found / n_positives) if n_positives > 0 else 0.0

        results[f"budget_{b}"] = {
            "budget": float(b),
            "k": int(k),
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
        }

    return results


def append_result(
    result_dict: Dict[str, Any],
    path: str = "results/results.csv",
) -> None:
    """Appends an evaluation result dictionary as a single row to a CSV file.

    Automatically creates parent directories and initializes the CSV file with
    appropriate column headers if it does not already exist.

    Args:
        result_dict: Dictionary containing experiment identifiers and metric scores.
        path: Destination file path for results.csv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0

    # Flatten any nested dictionaries if present (e.g. alert budgets)
    flat_record = {}
    for k, v in result_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_record[f"{k}_{sub_k}"] = sub_v
        else:
            flat_record[k] = v

    fieldnames = list(flat_record.keys())

    if file_exists:
        # Read existing headers to ensure consistent column ordering and append new fields if needed
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, [])
        for field in fieldnames:
            if field not in existing_headers:
                existing_headers.append(field)
        final_fields = existing_headers
    else:
        final_fields = fieldnames

    # If the file already exists but has missing columns compared to final_fields, re-write or append
    if file_exists:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_fields, extrasaction="ignore")
            writer.writerow(flat_record)
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_fields)
            writer.writeheader()
            writer.writerow(flat_record)

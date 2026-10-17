import os
import sys
import tempfile
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import evaluate_scores, best_f1_threshold, alert_budget_metrics, append_result


def test_perfect_scorer():
    """Checks that a perfect ranker/classifier achieves PR-AUC=1.0 and ROC-AUC=1.0."""
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.85, 0.95]

    res = evaluate_scores(y_true, y_score, threshold=0.5, model_name="perfect", split_name="test")

    assert res["n_rows"] == 5
    assert res["n_positives"] == 2
    assert res["prevalence"] == 0.4
    assert res["pr_auc"] == 1.0, f"Expected 1.0, got {res['pr_auc']}"
    assert res["roc_auc"] == 1.0, f"Expected 1.0, got {res['roc_auc']}"
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["tn"] == 3
    print("PASS: test_perfect_scorer")


def test_random_scorer():
    """Checks that an uninformative random scorer has PR-AUC approximately equal to prevalence."""
    rng = np.random.default_rng(42)
    n = 20000
    p = 0.05  # 5% prevalence
    y_true = rng.binomial(1, p, size=n)
    y_score = rng.uniform(0.0, 1.0, size=n)

    res = evaluate_scores(y_true, y_score, threshold=0.5)

    # In a random classifier:
    # ROC-AUC ~ 0.50
    # PR-AUC ~ prevalence
    assert abs(res["roc_auc"] - 0.50) < 0.03, f"Expected ROC-AUC near 0.5, got {res['roc_auc']}"
    assert abs(res["pr_auc"] - res["prevalence"]) < 0.02, (
        f"Expected PR-AUC near prevalence {res['prevalence']:.4f}, got {res['pr_auc']:.4f}"
    )
    print("PASS: test_random_scorer")


def test_best_f1_threshold():
    """Checks that best_f1_threshold finds a threshold that maximizes F1."""
    # Data with separated distributions where 0.5 might not be optimal
    y_true = [0] * 90 + [1] * 10
    # Negatives have scores 0.01 to 0.35, positives have scores 0.30 to 0.90
    y_score = list(np.linspace(0.01, 0.35, 90)) + list(np.linspace(0.30, 0.90, 10))

    threshold = best_f1_threshold(y_true, y_score)
    assert 0.20 <= threshold <= 0.40, f"Unexpected threshold {threshold}"

    res_opt = evaluate_scores(y_true, y_score, threshold=threshold)
    res_default = evaluate_scores(y_true, y_score, threshold=0.5)

    # Optimal threshold should yield at least as high F1 as default 0.5
    assert res_opt["f1"] >= res_default["f1"], (
        f"Optimal F1 {res_opt['f1']} should be >= default F1 {res_default['f1']}"
    )
    print(f"PASS: test_best_f1_threshold (opt_thresh={threshold:.4f}, F1={res_opt['f1']:.4f} vs def={res_default['f1']:.4f})")


def test_alert_budget_metrics():
    """Tests alert budget metric calculations on a precise hand-made example."""
    # 10 transactions, exactly 2 positives
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Budgets: 10% (1 alert), 20% (2 alerts), 50% (5 alerts)
    metrics = alert_budget_metrics(y_true, y_score, budgets=(0.1, 0.2, 0.5))

    # Budget 0.1 -> top 1 alert (score 1.0, label 1)
    b10 = metrics["budget_0.1"]
    assert b10["k"] == 1
    assert b10["precision_at_k"] == 1.0  # 1 found / 1 alert
    assert b10["recall_at_k"] == 0.5     # 1 found / 2 total positives

    # Budget 0.2 -> top 2 alerts (scores 1.0, 0.9, both label 1)
    b20 = metrics["budget_0.2"]
    assert b20["k"] == 2
    assert b20["precision_at_k"] == 1.0  # 2 found / 2 alerts
    assert b20["recall_at_k"] == 1.0     # 2 found / 2 total positives

    # Budget 0.5 -> top 5 alerts (scores 1.0, 0.9, 0.8, 0.7, 0.6; labels 1, 1, 0, 0, 0)
    b50 = metrics["budget_0.5"]
    assert b50["k"] == 5
    assert b50["precision_at_k"] == 0.4  # 2 found / 5 alerts
    assert b50["recall_at_k"] == 1.0     # 2 found / 2 total positives

    print("PASS: test_alert_budget_metrics")


def test_single_class_edge_case():
    """Checks that single-class arrays do not crash evaluate_scores."""
    # All negatives
    res_neg = evaluate_scores([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(res_neg["pr_auc"])
    assert np.isnan(res_neg["roc_auc"])
    assert res_neg["tn"] == 3
    assert res_neg["tp"] == 0

    # All positives
    res_pos = evaluate_scores([1, 1, 1], [0.8, 0.9, 0.95])
    assert np.isnan(res_pos["pr_auc"])
    assert np.isnan(res_pos["roc_auc"])
    assert res_pos["tp"] == 3
    assert res_pos["tn"] == 0
    print("PASS: test_single_class_edge_case")


def test_append_result():
    """Checks that append_result writes headers and multiple experiment rows cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = os.path.join(tmp_dir, "test_results.csv")
        r1 = {"model": "xgboost", "split": "test", "f1": 0.85, "pr_auc": 0.78}
        r2 = {"model": "lightgbm", "split": "test", "f1": 0.86, "pr_auc": 0.80}

        append_result(r1, path=csv_file)
        append_result(r2, path=csv_file)

        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert list(df["model"]) == ["xgboost", "lightgbm"]
        assert list(df["f1"]) == [0.85, 0.86]
    print("PASS: test_append_result")


if __name__ == "__main__":
    print("Running evaluate module unit tests...")
    test_perfect_scorer()
    test_random_scorer()
    test_best_f1_threshold()
    test_alert_budget_metrics()
    test_single_class_edge_case()
    test_append_result()
    print("All evaluate module tests passed successfully!")

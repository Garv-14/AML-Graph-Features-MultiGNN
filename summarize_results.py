"""Summarise tabular baseline results from results/results.csv.

Reads results.csv, groups by experiment (temporal vs random split), and computes:
- PR-AUC and Lift (PR-AUC / test prevalence), ROC-AUC, F1, Precision, Recall (mean +/- std)
- Test prevalence on its own line for each experiment
- Side-by-side comparison table of tabular_temporal and tabular_random
- Alert budget metrics table: Precision@k and Recall@k for 0.1%, 0.5%, 1%
- Mean training time per model in seconds

Prints the tables and saves the markdown to results/summary_tabular.md.

Usage
-----
  python summarize_results.py
  python summarize_results.py --results_csv path/to/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "xgboost",
    "isolation_forest",
    "lightgbm_undersample",
]

BUDGET_KEYS = [
    ("budget_0.001", "0.1%"),
    ("budget_0.005", "0.5%"),
    ("budget_0.01", "1.0%"),
]


def _format_mean_std(mean_val: float, std_val: float, decimals: int = 4) -> str:
    """Format as 'X.XXXX +/- Y.YYYY'."""
    if np.isnan(mean_val):
        return "   nan   "
    return f"{mean_val:.{decimals}f} +/- {std_val:.{decimals}f}"


def build_split_summary(subset: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Build summary table and prevalence info for one split kind."""
    if subset.empty:
        return "No results found.", pd.DataFrame()

    prevalence_vals = pd.to_numeric(subset["te_prevalence"], errors="coerce").dropna()
    mean_prev = float(prevalence_vals.mean()) if not prevalence_vals.empty else 0.0
    n_pos = int(pd.to_numeric(subset["te_n_positives"], errors="coerce").iloc[0]) if "te_n_positives" in subset.columns else 0
    n_rows = int(pd.to_numeric(subset["te_n_rows"], errors="coerce").iloc[0]) if "te_n_rows" in subset.columns else 0

    prev_line = f"**Test prevalence**: {mean_prev:.6f} ({mean_prev * 100:.3f}% | {n_pos:,} positives / {n_rows:,} transactions)"

    rows = []
    for model in MODEL_ORDER:
        mdf = subset[subset["model_name"] == model]
        if mdf.empty:
            continue
        row: dict = {"model": model}

        pr_auc = pd.to_numeric(mdf["te_pr_auc"], errors="coerce").dropna()
        pr_mean = float(pr_auc.mean())
        pr_std = float(pr_auc.std(ddof=0))
        row["PR-AUC"] = _format_mean_std(pr_mean, pr_std, 4)

        if mean_prev > 0:
            lift_vals = pr_auc / mean_prev
            row["Lift"] = _format_mean_std(float(lift_vals.mean()), float(lift_vals.std(ddof=0)), 2)
        else:
            row["Lift"] = "   n/a   "

        for col, label in [
            ("te_roc_auc", "ROC-AUC"),
            ("te_f1", "F1"),
            ("te_precision", "Precision"),
            ("te_recall", "Recall"),
        ]:
            if col in mdf.columns:
                vals = pd.to_numeric(mdf[col], errors="coerce").dropna()
                row[label] = _format_mean_std(float(vals.mean()), float(vals.std(ddof=0)), 4)
            else:
                row[label] = "   n/a   "

        rows.append(row)

    df_res = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
    return prev_line, df_res


def build_side_by_side_table(df_temporal: pd.DataFrame, df_random: pd.DataFrame) -> pd.DataFrame:
    """Build side-by-side comparison table of key metrics."""
    rows = []
    prev_temp = pd.to_numeric(df_temporal["te_prevalence"], errors="coerce").mean() if not df_temporal.empty else 0.0
    prev_rand = pd.to_numeric(df_random["te_prevalence"], errors="coerce").mean() if not df_random.empty else 0.0

    for model in MODEL_ORDER:
        m_temp = df_temporal[df_temporal["model_name"] == model]
        m_rand = df_random[df_random["model_name"] == model]
        if m_temp.empty and m_rand.empty:
            continue

        row: dict = {"model": model}

        # Temporal
        if not m_temp.empty:
            p_t = pd.to_numeric(m_temp["te_pr_auc"], errors="coerce").dropna()
            row["Temporal PR-AUC"] = _format_mean_std(float(p_t.mean()), float(p_t.std(ddof=0)), 4)
            row["Temporal Lift"] = _format_mean_std(float((p_t / prev_temp).mean()), float((p_t / prev_temp).std(ddof=0)), 2) if prev_temp > 0 else "n/a"
            f_t = pd.to_numeric(m_temp["te_f1"], errors="coerce").dropna()
            row["Temporal F1"] = _format_mean_std(float(f_t.mean()), float(f_t.std(ddof=0)), 4)
        else:
            row["Temporal PR-AUC"] = "n/a"
            row["Temporal Lift"] = "n/a"
            row["Temporal F1"] = "n/a"

        # Random
        if not m_rand.empty:
            p_r = pd.to_numeric(m_rand["te_pr_auc"], errors="coerce").dropna()
            row["Random PR-AUC"] = _format_mean_std(float(p_r.mean()), float(p_r.std(ddof=0)), 4)
            row["Random Lift"] = _format_mean_std(float((p_r / prev_rand).mean()), float((p_r / prev_rand).std(ddof=0)), 2) if prev_rand > 0 else "n/a"
            f_r = pd.to_numeric(m_rand["te_f1"], errors="coerce").dropna()
            row["Random F1"] = _format_mean_std(float(f_r.mean()), float(f_r.std(ddof=0)), 4)
        else:
            row["Random PR-AUC"] = "n/a"
            row["Random Lift"] = "n/a"
            row["Random F1"] = "n/a"

        rows.append(row)

    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def build_budget_table(subset: pd.DataFrame) -> pd.DataFrame:
    """Build alert budget metrics table: P@k and R@k for 0.1%, 0.5%, 1%."""
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for model in MODEL_ORDER:
        mdf = subset[subset["model_name"] == model]
        if mdf.empty:
            continue
        row: dict = {"model": model}
        for bkey, label in BUDGET_KEYS:
            p_col = f"te_{bkey}_precision_at_k"
            r_col = f"te_{bkey}_recall_at_k"
            if p_col in mdf.columns:
                p_vals = pd.to_numeric(mdf[p_col], errors="coerce").dropna()
                row[f"P@{label}"] = f"{p_vals.mean():.4f}" if not p_vals.empty else "n/a"
            else:
                row[f"P@{label}"] = "n/a"

            if r_col in mdf.columns:
                r_vals = pd.to_numeric(mdf[r_col], errors="coerce").dropna()
                row[f"R@{label}"] = f"{r_vals.mean():.4f}" if not r_vals.empty else "n/a"
            else:
                row[f"R@{label}"] = "n/a"

        rows.append(row)

    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def build_timings_table(df_temporal: pd.DataFrame, df_random: pd.DataFrame) -> pd.DataFrame:
    """Build mean training time table per model in seconds."""
    rows = []
    for model in MODEL_ORDER:
        row: dict = {"model": model}
        m_temp = df_temporal[df_temporal["model_name"] == model]
        m_rand = df_random[df_random["model_name"] == model]
        if m_temp.empty and m_rand.empty:
            continue

        if not m_temp.empty and "train_time_s" in m_temp.columns:
            t_vals = pd.to_numeric(m_temp["train_time_s"], errors="coerce").dropna()
            row["Temporal Time (s)"] = f"{t_vals.mean():.1f}s" if not t_vals.empty else "n/a"
        else:
            row["Temporal Time (s)"] = "n/a"

        if not m_rand.empty and "train_time_s" in m_rand.columns:
            r_vals = pd.to_numeric(m_rand["train_time_s"], errors="coerce").dropna()
            row["Random Time (s)"] = f"{r_vals.mean():.1f}s" if not r_vals.empty else "n/a"
        else:
            row["Random Time (s)"] = "n/a"

        rows.append(row)

    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print and save summary of tabular baseline results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_csv",
        default=str(Path(__file__).parent / "results" / "results.csv"),
        help="Path to results.csv.",
    )
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    if not results_csv.is_file():
        print(f"ERROR: results.csv not found at {results_csv}")
        print("Run run_tabular.py first.")
        return

    df = pd.read_csv(results_csv)

    tabular_mask = df["experiment"].str.startswith("tabular_", na=False)
    df = df[tabular_mask]

    if df.empty:
        print("No tabular experiment rows found in results.csv.")
        return

    df_temp = df[df["experiment"] == "tabular_temporal"]
    df_rand = df[df["experiment"] == "tabular_random"]

    output_lines: list[str] = []

    def _emit(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    _emit("# Tabular Baseline Results Summary (Step 4b)")
    _emit()
    _emit("Evaluation metrics reported as mean +/- std across seeds. Decision threshold selected on validation PR curve.")
    _emit()

    # 1. Side by side table
    _emit("## Side-by-Side Comparison (Temporal vs. Random)")
    _emit()
    side_by_side = build_side_by_side_table(df_temp, df_rand)
    if not side_by_side.empty:
        _emit(side_by_side.to_markdown())
    else:
        _emit("*(no results available)*")
    _emit()

    # 2. Temporal Split Details
    _emit("## Tabular Temporal Split (`tabular_temporal`)")
    _emit()
    prev_line_t, temp_table = build_split_summary(df_temp)
    _emit(prev_line_t)
    _emit()
    if not temp_table.empty:
        _emit(temp_table.to_markdown())
    else:
        _emit("*(no results found)*")
    _emit()

    # 3. Random Split Details
    _emit("## Tabular Random Split (`tabular_random`)")
    _emit()
    prev_line_r, rand_table = build_split_summary(df_rand)
    _emit(prev_line_r)
    _emit()
    if not rand_table.empty:
        _emit(rand_table.to_markdown())
    else:
        _emit("*(no results found)*")
    _emit()

    # 4. Alert budget metrics
    _emit("## Alert Budget Metrics (Precision@k and Recall@k)")
    _emit()
    _emit("Operational compliance investigation capacity at top 0.1%, 0.5%, and 1.0% alerts.")
    _emit()
    _emit("### Temporal Split")
    _emit()
    budget_t = build_budget_table(df_temp)
    if not budget_t.empty:
        _emit(budget_t.to_markdown())
    else:
        _emit("*(no results found)*")
    _emit()

    _emit("### Random Split")
    _emit()
    budget_r = build_budget_table(df_rand)
    if not budget_r.empty:
        _emit(budget_r.to_markdown())
    else:
        _emit("*(no results found)*")
    _emit()

    # 5. Training time summary
    _emit("## Mean Training Time per Model (seconds)")
    _emit()
    timings_table = build_timings_table(df_temp, df_rand)
    if not timings_table.empty:
        _emit(timings_table.to_markdown())
    else:
        _emit("*(no results found)*")
    _emit()

    # Save to markdown
    out_path = Path(args.results_csv).parent / "summary_tabular.md"
    out_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nSummary saved to: {out_path}")


if __name__ == "__main__":
    main()

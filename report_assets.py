"""Shared reporting assets, figures, and tables generator for AML report.

Generates all figures and tables used in notebooks/01_eda.ipynb and
notebooks/02_results.ipynb, as well as results/final_tables.md.
All figures are saved to results/figures/ at 200 dpi with the 'report_' prefix.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

# Reused project modules
from data_access import get_data_dir, load_formatted, load_split
from explain_checks import recover_and_verify_category_maps
from run_tabular import load_and_build_features

# Base directories
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDS_DIR = RESULTS_DIR / "preds"
SPLITS_DIR = BASE_DIR / "splits"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Consistent model colors
MODEL_COLORS: dict[str, str] = {
    "logistic_regression": "#4e79a7",       # Steel Blue
    "random_forest": "#f28e2b",             # Warm Orange
    "xgboost": "#59a14f",                   # Forest Green
    "lightgbm_undersample": "#e15759",      # Coral Red
    "isolation_forest": "#b07aa1",          # Muted Purple
    "lightgbm": "#76b7b2",                  # Teal (excluded baseline)
    "DecisionTree (depth 3)": "#edc948",    # Yellow/Gold
    "Simple Rule": "#9c755f",               # Brown
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm_undersample": "LightGBM (Undersample)",
    "isolation_forest": "Isolation Forest",
}

FEATURE_SET_COLORS: dict[str, str] = {
    "tabular": "#4e79a7",
    "tab_dyn": "#e15759",
    "tab_dyn_static": "#59a14f",
}

FEATURE_SET_DISPLAY_NAMES: dict[str, str] = {
    "tabular": "Tabular",
    "tab_dyn": "Tab + Dyn",
    "tab_dyn_static": "Tab + Dyn + Stat",
}


# ==============================================================================
# SECTION 1: EDA ASSETS (notebooks/01_eda.ipynb)
# ==============================================================================

def eda_overview_table() -> pd.DataFrame:
    """1.a: Overview table: rows, accounts, days, laundering rows, and rate."""
    fmt_path = Path(get_data_dir()) / "Small_HI" / "formatted_transactions.csv"
    df = pd.read_csv(fmt_path, usecols=["from_id", "to_id", "Timestamp", "Is Laundering"])

    n_rows = int(len(df))
    n_accounts = int(len(np.unique(np.concatenate([df["from_id"].values, df["to_id"].values]))))
    n_days = int(np.floor(df["Timestamp"].max() / 86400.0) + 1)
    n_laundering = int(df["Is Laundering"].sum())
    laundering_rate = n_laundering / n_rows

    overview = pd.DataFrame([
        {
            "Metric": "Total Transactions (Rows)",
            "Value": f"{n_rows:,}",
        },
        {
            "Metric": "Distinct Accounts",
            "Value": f"{n_accounts:,}",
        },
        {
            "Metric": "Time Span (Days)",
            "Value": str(n_days),
        },
        {
            "Metric": "Laundering Transactions",
            "Value": f"{n_laundering:,}",
        },
        {
            "Metric": "Laundering Prevalence Rate",
            "Value": f"{laundering_rate:.6f} ({laundering_rate * 100:.4f}%)",
        },
    ])
    return overview


def eda_daily_laundering_plot(
    save_path: Path | str = FIGURES_DIR / "report_eda_daily_laundering.png",
) -> pd.DataFrame:
    """1.b: Daily laundering rate (line) and daily transaction volume (bars) with split bounds."""
    fmt_path = Path(get_data_dir()) / "Small_HI" / "formatted_transactions.csv"
    df = pd.read_csv(fmt_path, usecols=["Timestamp", "Is Laundering"])

    df["day"] = (df["Timestamp"] // 86400).astype(int)
    daily = (
        df.groupby("day")
        .agg(
            total_transactions=("Is Laundering", "count"),
            laundering_transactions=("Is Laundering", "sum"),
        )
        .reset_index()
    )
    daily["laundering_rate_pct"] = (
        daily["laundering_transactions"] / daily["total_transactions"] * 100.0
    )

    # Boundaries from Small_HI_split_summary.json / export_splits:
    # Train: days 0..5, Val: days 6..7, Test: days 8..17
    # Boundary lines at 5.5 and 7.5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel 1: Laundering rate (%)
    ax1.plot(
        daily["day"],
        daily["laundering_rate_pct"],
        marker="o",
        color="#e15759",
        linewidth=2,
        markersize=6,
        label="Laundering Rate (%)",
    )
    ax1.axvline(x=5.5, color="#4e79a7", linestyle="--", linewidth=1.5)
    ax1.axvline(x=7.5, color="#59a14f", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("Laundering Rate (%)", fontsize=11)
    ax1.set_title("Laundering Prevalence and Daily Volume Across Temporal Splits", fontsize=13, pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Annotate partitions on top panel
    ax1.text(2.5, ax1.get_ylim()[1] * 0.88, "Train Split (Days 0-5)", ha="center", fontsize=10, fontweight="bold", color="#333333")
    ax1.text(6.5, ax1.get_ylim()[1] * 0.88, "Val (6-7)", ha="center", fontsize=10, fontweight="bold", color="#333333")
    ax1.text(12.5, ax1.get_ylim()[1] * 0.88, "Test Split (Days 8-17)", ha="center", fontsize=10, fontweight="bold", color="#333333")

    # Panel 2: Total transactions
    ax2.bar(
        daily["day"],
        daily["total_transactions"] / 1000.0,
        color="#4e79a7",
        alpha=0.75,
        edgecolor="#2b4c6f",
        label="Transactions (thousands)",
    )
    ax2.axvline(x=5.5, color="#4e79a7", linestyle="--", linewidth=1.5)
    ax2.axvline(x=7.5, color="#59a14f", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Day (Chronological Stream, 0-indexed)", fontsize=11)
    ax2.set_ylabel("Transactions (k)", fontsize=11)
    ax2.set_xticks(range(int(daily["day"].max()) + 1))
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return daily


def eda_payment_format_table_and_plot(
    save_path: Path | str = FIGURES_DIR / "report_eda_payment_format.png",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1.c: Payment format distribution table & plot, plus top 8 currencies table."""
    inv_payment_format, inv_currency, verified = recover_and_verify_category_maps()
    if not verified:
        raise ValueError("Category mapping could not be verified against raw data!")

    fmt_path = Path(get_data_dir()) / "Small_HI" / "formatted_transactions.csv"
    df = pd.read_csv(fmt_path, usecols=["Payment Format", "Sent Currency", "Is Laundering"])

    n_pos_total = int(df["Is Laundering"].sum())
    n_neg_total = int((df["Is Laundering"] == 0).sum())

    # --- Payment format table ---
    pf_rows = []
    for code, name in sorted(inv_payment_format.items()):
        sub = df[df["Payment Format"] == code]
        n_rows = len(sub)
        n_pos = int(sub["Is Laundering"].sum())
        n_neg = n_rows - n_pos
        rate = (n_pos / n_rows * 100.0) if n_rows > 0 else 0.0
        share_pos = (n_pos / n_pos_total * 100.0) if n_pos_total > 0 else 0.0
        share_neg = (n_neg / n_neg_total * 100.0) if n_neg_total > 0 else 0.0
        pf_rows.append({
            "Code": code,
            "Payment Format": name,
            "Rows": n_rows,
            "Laundering Rows": n_pos,
            "Laundering Rate (%)": round(rate, 4),
            "Share of Positives (%)": round(share_pos, 2),
            "Share of Negatives (%)": round(share_neg, 2),
        })
    df_pf = pd.DataFrame(pf_rows).sort_values("Rows", ascending=False).reset_index(drop=True)

    # --- Payment format plot (share of positives vs negatives) ---
    plot_df = df_pf.sort_values("Share of Positives (%)", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df))
    width = 0.35

    ax.bar(x - width / 2, plot_df["Share of Positives (%)"], width, label="Share of All Positives (%)", color="#e15759")
    ax.bar(x + width / 2, plot_df["Share of Negatives (%)"], width, label="Share of All Negatives (%)", color="#4e79a7")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Payment Format"], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Payment Format Distribution: Positives vs. Negatives", fontsize=13, pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    # --- Top 8 Currencies (Sent Currency) table ---
    top_curr_codes = df["Sent Currency"].value_counts().head(8).index.tolist()
    curr_rows = []
    for code in top_curr_codes:
        name = inv_currency.get(code, f"Currency_{code}")
        sub = df[df["Sent Currency"] == code]
        n_rows = len(sub)
        n_pos = int(sub["Is Laundering"].sum())
        n_neg = n_rows - n_pos
        rate = (n_pos / n_rows * 100.0) if n_rows > 0 else 0.0
        share_pos = (n_pos / n_pos_total * 100.0) if n_pos_total > 0 else 0.0
        share_neg = (n_neg / n_neg_total * 100.0) if n_neg_total > 0 else 0.0
        curr_rows.append({
            "Code": code,
            "Currency": name,
            "Rows": n_rows,
            "Laundering Rows": n_pos,
            "Laundering Rate (%)": round(rate, 4),
            "Share of Positives (%)": round(share_pos, 2),
            "Share of Negatives (%)": round(share_neg, 2),
        })
    df_curr = pd.DataFrame(curr_rows)

    return df_pf, df_curr


def eda_amounts_table_and_plot(
    save_path: Path | str = FIGURES_DIR / "report_eda_amounts.png",
    neg_sample_size: int = 200000,
    seed: int = 0,
) -> pd.DataFrame:
    """1.d: Histogram of log10(Amount Sent) (density-norm) and percentile table."""
    fmt_path = Path(get_data_dir()) / "Small_HI" / "formatted_transactions.csv"
    df = pd.read_csv(fmt_path, usecols=["Amount Sent", "Is Laundering"])

    pos_amounts = df[df["Is Laundering"] == 1]["Amount Sent"].values
    neg_all = df[df["Is Laundering"] == 0]["Amount Sent"].values

    rng = np.random.RandomState(seed)
    neg_sample = rng.choice(neg_all, size=min(neg_sample_size, len(neg_all)), replace=False)

    # Compute log10 amounts (handling values > 0)
    pos_log10 = np.log10(np.maximum(pos_amounts, 1e-4))
    neg_log10 = np.log10(np.maximum(neg_sample, 1e-4))

    # Summary table (original amounts and log10)
    summary_rows = [
        {
            "Class": f"Positives (All N={len(pos_amounts):,})",
            "Median Amount": round(float(np.median(pos_amounts)), 2),
            "99th Percentile Amount": round(float(np.percentile(pos_amounts, 99)), 2),
            "Median log10(Amount)": round(float(np.median(pos_log10)), 3),
            "99th Percentile log10(Amount)": round(float(np.percentile(pos_log10, 99)), 3),
        },
        {
            "Class": f"Negatives (Sample N={len(neg_sample):,})",
            "Median Amount": round(float(np.median(neg_sample)), 2),
            "99th Percentile Amount": round(float(np.percentile(neg_sample, 99)), 2),
            "Median log10(Amount)": round(float(np.median(neg_log10)), 3),
            "99th Percentile log10(Amount)": round(float(np.percentile(neg_log10, 99)), 3),
        },
    ]
    df_summary = pd.DataFrame(summary_rows)

    # Plot density-normalized histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1, 8, 45)

    ax.hist(pos_log10, bins=bins, density=True, alpha=0.55, color="#e15759", label=f"Positives (N={len(pos_amounts):,})", edgecolor="none")
    ax.hist(neg_log10, bins=bins, density=True, alpha=0.45, color="#4e79a7", label=f"Negatives Sample (N={len(neg_sample):,})", edgecolor="none")

    # Mark medians
    ax.axvline(np.median(pos_log10), color="#b22222", linestyle="--", linewidth=1.5, label=f"Pos Median ({np.median(pos_log10):.2f})")
    ax.axvline(np.median(neg_log10), color="#1c3d5a", linestyle="--", linewidth=1.5, label=f"Neg Median ({np.median(neg_log10):.2f})")

    ax.set_xlabel("log10(Amount Sent)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Density-Normalized Distribution of log10(Amount Sent)", fontsize=13, pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return df_summary


def eda_time_patterns_plot(
    save_path: Path | str = FIGURES_DIR / "report_eda_hourly_weekly.png",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1.e: Laundering rate plots by hour of day and day of week."""
    fmt_path = Path(get_data_dir()) / "Small_HI" / "formatted_transactions.csv"
    df = pd.read_csv(fmt_path, usecols=["Timestamp", "Is Laundering"])

    # Epoch weekday offset is 3 (Thursday)
    ts = df["Timestamp"].astype("int64")
    df["hour_of_day"] = ((ts % 86400) // 3600).astype(int)
    df["day_of_week"] = ((ts // 86400 + 3) % 7).astype(int)

    # Hourly stats
    hourly = (
        df.groupby("hour_of_day")
        .agg(total=("Is Laundering", "count"), laundering=("Is Laundering", "sum"))
        .reset_index()
    )
    hourly["laundering_rate_pct"] = hourly["laundering"] / hourly["total"] * 100.0

    # Day of week stats (0=Monday, 6=Sunday)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = (
        df.groupby("day_of_week")
        .agg(total=("Is Laundering", "count"), laundering=("Is Laundering", "sum"))
        .reset_index()
    )
    dow["laundering_rate_pct"] = dow["laundering"] / dow["total"] * 100.0
    dow["day_name"] = [dow_names[i] for i in dow["day_of_week"]]

    # 2-Panel Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Hour of Day
    ax1.plot(
        hourly["hour_of_day"],
        hourly["laundering_rate_pct"],
        marker="o",
        color="#e15759",
        linewidth=2,
        markersize=5,
    )
    ax1.set_xlabel("Hour of Day (UTC)", fontsize=11)
    ax1.set_ylabel("Laundering Rate (%)", fontsize=11)
    ax1.set_title("Laundering Rate by Hour of Day", fontsize=12, pad=10)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Day of Week
    ax2.bar(
        dow["day_name"],
        dow["laundering_rate_pct"],
        color="#4e79a7",
        alpha=0.8,
        edgecolor="#2b4c6f",
    )
    ax2.set_xlabel("Day of Week", fontsize=11)
    ax2.set_ylabel("Laundering Rate (%)", fontsize=11)
    ax2.set_title("Laundering Rate by Day of Week", fontsize=12, pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return hourly, dow


def eda_account_history(
    save_path: Path | str = FIGURES_DIR / "report_eda_pair_prior_cnt_daily.png",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1.f: Share of no sender/receiver history in splits, and daily pair_prior_cnt==0 plot."""
    data_dir = Path(get_data_dir()) / "Small_HI"
    dyn_path = data_dir / "graph_dynamic.parquet"
    df_dyn = pd.read_parquet(dyn_path, columns=["has_sender_history", "has_receiver_history", "pair_prior_cnt"])

    fmt_path = data_dir / "formatted_transactions.csv"
    df_ts = pd.read_csv(fmt_path, usecols=["Timestamp"])
    df_dyn["day"] = (df_ts["Timestamp"] // 86400).astype(int)

    # Load temporal split indices
    sp = np.load(SPLITS_DIR / "Small_HI_split.npz")
    splits = {
        "Train": sp["tr_inds"],
        "Validation": sp["val_inds"],
        "Test": sp["te_inds"],
    }

    cov_rows = []
    for split_name, inds in splits.items():
        sub = df_dyn.iloc[inds]
        n = len(sub)
        no_s = int((sub["has_sender_history"] == 0).sum())
        no_r = int((sub["has_receiver_history"] == 0).sum())
        cov_rows.append({
            "Split": split_name,
            "Total Rows": n,
            "No Sender History Rows": no_s,
            "No Sender History (%)": round(no_s / n * 100.0, 2),
            "No Receiver History Rows": no_r,
            "No Receiver History (%)": round(no_r / n * 100.0, 2),
        })
    df_coverage = pd.DataFrame(cov_rows)

    # Daily share of pair_prior_cnt == 0
    daily_pair = (
        df_dyn.groupby("day")
        .agg(
            total=("pair_prior_cnt", "count"),
            zero_prior=("pair_prior_cnt", lambda s: int((s == 0.0).sum())),
        )
        .reset_index()
    )
    daily_pair["zero_prior_share_pct"] = daily_pair["zero_prior"] / daily_pair["total"] * 100.0

    # Line plot with split boundaries marked
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        daily_pair["day"],
        daily_pair["zero_prior_share_pct"],
        marker="s",
        color="#f28e2b",
        linewidth=2,
        markersize=6,
        label="First-Time Directed Pair (pair_prior_cnt == 0)",
    )
    ax.axvline(x=5.5, color="#4e79a7", linestyle="--", linewidth=1.5, label="Train/Val Boundary")
    ax.axvline(x=7.5, color="#59a14f", linestyle="--", linewidth=1.5, label="Val/Test Boundary")

    ax.set_xlabel("Day (Chronological Stream)", fontsize=11)
    ax.set_ylabel("Share of Transactions (%)", fontsize=11)
    ax.set_title("Share of First-Time Pair Transactions per Day Across Split Boundaries", fontsize=13, pad=12)
    ax.set_xticks(range(int(daily_pair["day"].max()) + 1))
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return df_coverage, daily_pair


def eda_pair_prior_laundering_rate() -> pd.DataFrame:
    """1.g: Laundering rate for pair_prior_cnt == 0 vs > 0 per split."""
    data_dir = Path(get_data_dir()) / "Small_HI"
    df_dyn = pd.read_parquet(data_dir / "graph_dynamic.parquet", columns=["pair_prior_cnt"])
    df_fmt = pd.read_csv(data_dir / "formatted_transactions.csv", usecols=["Is Laundering"])
    y_all = df_fmt["Is Laundering"].values.astype(int)

    sp = np.load(SPLITS_DIR / "Small_HI_split.npz")
    splits = {
        "Train": sp["tr_inds"],
        "Validation": sp["val_inds"],
        "Test": sp["te_inds"],
    }

    rows = []
    for split_name, inds in splits.items():
        pairs = df_dyn["pair_prior_cnt"].iloc[inds].values
        y_split = y_all[inds]

        mask_0 = pairs == 0.0
        mask_gt = pairs > 0.0

        n0 = int(mask_0.sum())
        p0 = int((y_split[mask_0] == 1).sum())
        rate0 = (p0 / n0 * 100.0) if n0 > 0 else 0.0

        ngt = int(mask_gt.sum())
        pgt = int((y_split[mask_gt] == 1).sum())
        rategt = (pgt / ngt * 100.0) if ngt > 0 else 0.0

        enrichment = rate0 / max(rategt, 1e-9)

        rows.append({
            "Split": split_name,
            "Condition": "pair_prior_cnt == 0",
            "Total Rows": n0,
            "Laundering Rows": p0,
            "Laundering Rate (%)": round(rate0, 4),
            "Enrichment vs Prior": round(enrichment, 2),
        })
        rows.append({
            "Split": split_name,
            "Condition": "pair_prior_cnt > 0",
            "Total Rows": ngt,
            "Laundering Rows": pgt,
            "Laundering Rate (%)": round(rategt, 4),
            "Enrichment vs Prior": 1.0,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# SECTION 2: RESULTS ASSETS (notebooks/02_results.ipynb)
# ==============================================================================

def results_final_tables(
    md_out_path: Path | str = RESULTS_DIR / "final_tables.md",
) -> dict[str, pd.DataFrame]:
    """2.a: Tables from results/results.csv for each experiment. Excludes plain lightgbm."""
    df_res = pd.read_csv(RESULTS_DIR / "results.csv")

    # Exclude plain lightgbm
    df_res = df_res[df_res["model_name"] != "lightgbm"].copy()

    # Calculate lift
    df_res["lift"] = df_res["te_pr_auc"] / df_res["te_prevalence"]

    experiments = [
        "tabular_temporal",
        "tabular_random",
        "ablation_tab_dyn_temporal",
        "ablation_tab_dyn_random",
        "ablation_tab_dyn_static_temporal",
        "ablation_tab_dyn_static_random",
    ]

    metric_cols = ["te_pr_auc", "lift", "te_roc_auc", "te_f1", "te_precision", "te_recall"]
    display_cols = ["PR-AUC", "Lift", "ROC-AUC", "F1", "Precision", "Recall"]

    exp_tables: dict[str, pd.DataFrame] = {}
    md_lines: list[str] = [
        "# Benchmark Results across Experiments ( IBM AML Small_HI )",
        "",
        "Aggregated across seeds (mean ± std). Plain `lightgbm` is excluded from reporting "
        "because it experienced single-round early stopping and severe score collapse (see CHANGES.md Step 4c).",
        "",
    ]

    for exp in experiments:
        sub = df_res[df_res["experiment"] == exp]
        if sub.empty:
            continue

        models = sub["model_name"].unique()
        table_rows = []
        for m in models:
            m_sub = sub[sub["model_name"] == m]
            n_seeds = len(m_sub)
            row_dict = {
                "Model": MODEL_DISPLAY_NAMES.get(m, m),
                "Seeds": n_seeds,
            }
            for col, dcol in zip(metric_cols, display_cols):
                mean_v = m_sub[col].mean()
                std_v = m_sub[col].std(ddof=1) if n_seeds > 1 else 0.0
                if dcol == "Lift":
                    row_dict[dcol] = f"{mean_v:.1f} ± {std_v:.1f}"
                else:
                    row_dict[dcol] = f"{mean_v:.4f} ± {std_v:.4f}"
            table_rows.append(row_dict)

        df_table = pd.DataFrame(table_rows)
        exp_tables[exp] = df_table

        md_lines.append(f"## Experiment: `{exp}`")
        md_lines.append("")
        md_lines.append(df_table.to_markdown(index=False))
        md_lines.append("")

    md_lines.append("---")
    md_lines.append(
        "**Footnote**: Plain `lightgbm` with `scale_pos_weight = negatives/positives` stopped prematurely "
        "at round 1 due to validation logloss conflict and is superseded by `lightgbm_undersample`."
    )
    md_lines.append("")

    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return exp_tables


def results_lift_by_feature_set_plot(
    save_path: Path | str = FIGURES_DIR / "report_results_lift_by_features.png",
) -> pd.DataFrame:
    """2.b: Grouped bars of LIFT per model for feature sets in 2 panels (temporal, random)."""
    df_res = pd.read_csv(RESULTS_DIR / "results.csv")
    df_res["lift"] = df_res["te_pr_auc"] / df_res["te_prevalence"]

    models = ["logistic_regression", "random_forest", "xgboost", "lightgbm_undersample"]
    feature_sets = ["tabular", "tab_dyn", "tab_dyn_static"]

    exp_map = {
        ("temporal", "tabular"): "tabular_temporal",
        ("temporal", "tab_dyn"): "ablation_tab_dyn_temporal",
        ("temporal", "tab_dyn_static"): "ablation_tab_dyn_static_temporal",
        ("random", "tabular"): "tabular_random",
        ("random", "tab_dyn"): "ablation_tab_dyn_random",
        ("random", "tab_dyn_static"): "ablation_tab_dyn_static_random",
    }

    records = []
    for split in ["temporal", "random"]:
        for fset in feature_sets:
            exp_name = exp_map[(split, fset)]
            sub = df_res[df_res["experiment"] == exp_name]
            for m in models:
                m_sub = sub[sub["model_name"] == m]
                mean_lift = m_sub["lift"].mean()
                std_lift = m_sub["lift"].std(ddof=1) if len(m_sub) > 1 else 0.0
                records.append({
                    "split": split,
                    "feature_set": fset,
                    "model": m,
                    "model_display": MODEL_DISPLAY_NAMES.get(m, m),
                    "mean_lift": mean_lift,
                    "std_lift": std_lift,
                })
    df_lift = pd.DataFrame(records)

    # Plot 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def draw_panel(ax, split_name, title):
        sub_split = df_lift[df_lift["split"] == split_name]
        x = np.arange(len(models))
        width = 0.25
        offsets = [-width, 0, width]

        for i, fset in enumerate(feature_sets):
            sub_fset = sub_split[sub_split["feature_set"] == fset].set_index("model").loc[models]
            ax.bar(
                x + offsets[i],
                sub_fset["mean_lift"],
                width,
                yerr=sub_fset["std_lift"],
                capsize=4,
                label=FEATURE_SET_DISPLAY_NAMES[fset],
                color=FEATURE_SET_COLORS[fset],
                alpha=0.85,
                edgecolor="#333333",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in models], rotation=15, ha="right", fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    draw_panel(ax1, "temporal", "Temporal Split (Real-world Chronological)")
    ax1.set_ylabel("PR-AUC Lift (over Prevalence)", fontsize=11)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    draw_panel(ax2, "random", "Random Split (Uniform Permutation)")

    fig.suptitle("Test PR-AUC Lift Across Models and Feature Sets", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return df_lift


def results_temporal_vs_random_lift_plot(
    save_path: Path | str = FIGURES_DIR / "report_results_lift_inflation.png",
) -> pd.DataFrame:
    """2.c: Temporal vs Random LIFT and the inflation ratio (Random / Temporal)."""
    df_res = pd.read_csv(RESULTS_DIR / "results.csv")
    df_res["lift"] = df_res["te_pr_auc"] / df_res["te_prevalence"]

    models = ["logistic_regression", "random_forest", "xgboost", "lightgbm_undersample"]
    feature_sets = ["tabular", "tab_dyn", "tab_dyn_static"]

    records = []
    for fset in feature_sets:
        exp_temp = "tabular_temporal" if fset == "tabular" else f"ablation_{fset}_temporal"
        exp_rand = "tabular_random" if fset == "tabular" else f"ablation_{fset}_random"

        sub_temp = df_res[df_res["experiment"] == exp_temp]
        sub_rand = df_res[df_res["experiment"] == exp_rand]

        for m in models:
            temp_lift = sub_temp[sub_temp["model_name"] == m]["lift"].mean()
            rand_lift = sub_rand[sub_rand["model_name"] == m]["lift"].mean()
            ratio = rand_lift / max(temp_lift, 1e-9)

            records.append({
                "feature_set": fset,
                "model": m,
                "model_display": MODEL_DISPLAY_NAMES.get(m, m),
                "temporal_lift": round(temp_lift, 2),
                "random_lift": round(rand_lift, 2),
                "inflation_ratio": round(ratio, 2),
            })

    df_comp = pd.DataFrame(records)

    # 2 Panels: Panel 1 = Temporal vs Random Lift, Panel 2 = Inflation Ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Bar chart comparing Temporal vs Random for each combination
    labels = [f"{r['model_display']}\n({FEATURE_SET_DISPLAY_NAMES[r['feature_set']]})" for _, r in df_comp.iterrows()]
    y = np.arange(len(labels))
    height = 0.35

    ax1.barh(y - height / 2, df_comp["temporal_lift"], height, label="Temporal Lift", color="#4e79a7")
    ax1.barh(y + height / 2, df_comp["random_lift"], height, label="Random Lift", color="#e15759")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("Lift (PR-AUC / Test Prevalence)", fontsize=11)
    ax1.set_title("Temporal vs. Random Split Lift", fontsize=12)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # Panel 2: Inflation Ratio
    bars = ax2.barh(y, df_comp["inflation_ratio"], color="#59a14f", alpha=0.85, edgecolor="#2b4c6f")
    ax2.axvline(1.0, color="#b22222", linestyle="--", linewidth=1.5, label="No Inflation (1.0x)")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Inflation Ratio (Random Lift / Temporal Lift)", fontsize=11)
    ax2.set_title("Split Discrepancy (Leakage / Drift Factor)", fontsize=12)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    # Annotate ratios
    for bar, ratio in zip(bars, df_comp["inflation_ratio"]):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{ratio:.2f}x", va="center", fontsize=8.5)

    fig.suptitle("Impact of Temporal vs. Random Partitioning on Model Performance", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return df_comp


def results_pr_curves_plot(
    save_path: Path | str = FIGURES_DIR / "report_results_pr_curves.png",
) -> dict[str, float]:
    """2.d: Precision-Recall curves for lightgbm_undersample (tabular vs tab_dyn, seed 0)."""
    df_tr, df_val, df_te = load_split("Small_HI", "temporal")
    y_te = df_te["Is Laundering"].values.astype(int)
    te_prev = float(y_te.mean())

    s_tab = np.load(PREDS_DIR / "lightgbm_undersample_temporal_seed0_te_scores.npy")
    s_dyn = np.load(PREDS_DIR / "ablation_tab_dyn_temporal_lightgbm_undersample_seed0_te_scores.npy")

    pr_tab = float(average_precision_score(y_te, s_tab))
    pr_dyn = float(average_precision_score(y_te, s_dyn))

    # Read results.csv to verify < 0.01
    df_res = pd.read_csv(RESULTS_DIR / "results.csv")
    r_tab = float(df_res[(df_res["experiment"] == "tabular_temporal") & (df_res["model_name"] == "lightgbm_undersample") & (df_res["seed"] == 0)]["te_pr_auc"].values[0])
    r_dyn = float(df_res[(df_res["experiment"] == "ablation_tab_dyn_temporal") & (df_res["model_name"] == "lightgbm_undersample") & (df_res["seed"] == 0)]["te_pr_auc"].values[0])

    if abs(pr_tab - r_tab) > 0.01 or abs(pr_dyn - r_dyn) > 0.01:
        raise ValueError(f"PR-AUC discrepancy exceeds 0.01 threshold! tab: {abs(pr_tab-r_tab)}, dyn: {abs(pr_dyn-r_dyn)}")

    print(f"Verified test PR-AUC: Tabular={pr_tab:.6f} (matches results.csv {r_tab:.6f}), Tab_dyn={pr_dyn:.6f} (matches results.csv {r_dyn:.6f})")

    p_tab, r_curve_tab, _ = precision_recall_curve(y_te, s_tab)
    p_dyn, r_curve_dyn, _ = precision_recall_curve(y_te, s_dyn)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(r_curve_dyn, p_dyn, color="#e15759", linewidth=2.2, label=f"Tab + Dyn (Seed 0, PR-AUC = {pr_dyn:.4f})")
    ax.plot(r_curve_tab, p_tab, color="#4e79a7", linewidth=2.0, label=f"Tabular Baseline (Seed 0, PR-AUC = {pr_tab:.4f})")
    ax.axhline(te_prev, color="#888888", linestyle="--", linewidth=1.5, label=f"Test Prevalence Baseline ({te_prev:.4f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves: LightGBM (Undersample) on Temporal Test Set", fontsize=13, pad=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return {
        "pr_auc_tabular": pr_tab,
        "pr_auc_tab_dyn": pr_dyn,
        "test_prevalence": te_prev,
        "matches_results_csv": True,
    }


def results_alert_budget_table_and_plot(
    save_path: Path | str = FIGURES_DIR / "report_results_alert_budgets.png",
) -> pd.DataFrame:
    """2.e: Alert-budget curves and table for rf, xgboost, lightgbm_undersample on tab_dyn temporal."""
    df_res = pd.read_csv(RESULTS_DIR / "results.csv")
    sub = df_res[df_res["experiment"] == "ablation_tab_dyn_temporal"]

    models = ["random_forest", "xgboost", "lightgbm_undersample"]
    budgets = ["0.001", "0.005", "0.01"]
    budget_pcts = [0.1, 0.5, 1.0]

    table_rows = []
    plot_records = []

    for m in models:
        m_sub = sub[sub["model_name"] == m]
        m_display = MODEL_DISPLAY_NAMES.get(m, m)
        for b, bpct in zip(budgets, budget_pcts):
            k_val = int(m_sub[f"te_budget_{b}_k"].values[0])
            p_mean = m_sub[f"te_budget_{b}_precision_at_k"].mean()
            p_std = m_sub[f"te_budget_{b}_precision_at_k"].std(ddof=1)
            r_mean = m_sub[f"te_budget_{b}_recall_at_k"].mean()
            r_std = m_sub[f"te_budget_{b}_recall_at_k"].std(ddof=1)

            table_rows.append({
                "Model": m_display,
                "Budget": f"{bpct}%",
                "Alerts (k)": f"{k_val:,}",
                "Precision@k (%)": f"{p_mean * 100:.2f} ± {p_std * 100:.2f}",
                "Recall@k (%)": f"{r_mean * 100:.2f} ± {r_std * 100:.2f}",
            })
            plot_records.append({
                "model": m,
                "model_display": m_display,
                "budget_pct": bpct,
                "k": k_val,
                "p_mean": p_mean * 100.0,
                "p_std": p_std * 100.0,
                "r_mean": r_mean * 100.0,
                "r_std": r_std * 100.0,
            })

    df_budget_table = pd.DataFrame(table_rows)
    df_plot = pd.DataFrame(plot_records)

    # Plot 2 panels: Precision@k vs Budget, Recall@k vs Budget
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for m in models:
        m_data = df_plot[df_plot["model"] == m]
        c = MODEL_COLORS[m]
        disp = MODEL_DISPLAY_NAMES[m]

        ax1.errorbar(
            m_data["budget_pct"],
            m_data["p_mean"],
            yerr=m_data["p_std"],
            marker="o",
            capsize=4,
            linewidth=2,
            color=c,
            label=disp,
        )
        ax2.errorbar(
            m_data["budget_pct"],
            m_data["r_mean"],
            yerr=m_data["r_std"],
            marker="s",
            capsize=4,
            linewidth=2,
            color=c,
            label=disp,
        )

    budget_labels = [f"{bpct}% ({int(sub[f'te_budget_{b}_k'].values[0]):,})" for b, bpct in zip(budgets, budget_pcts)]
    ax1.set_xlabel("Compliance Alert Budget (% of Test Rows)", fontsize=11)
    ax1.set_ylabel("Precision@k (%)", fontsize=11)
    ax1.set_title("Alert Precision by Inspection Budget (Tab + Dyn, Temporal)", fontsize=12, pad=10)
    ax1.set_xticks(budget_pcts)
    ax1.set_xticklabels(budget_labels)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_xlabel("Compliance Alert Budget (% of Test Rows)", fontsize=11)
    ax2.set_ylabel("Recall@k (%)", fontsize=11)
    ax2.set_title("Laundering Recall by Inspection Budget (Tab + Dyn, Temporal)", fontsize=12, pad=10)
    ax2.set_xticks(budget_pcts)
    ax2.set_xticklabels(budget_labels)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return df_budget_table


def results_simple_rule_check() -> pd.DataFrame:
    """2.f: Simple-rule checks on the full temporal test set compared to LightGBM and DecisionTree."""
    inv_pf, _, _ = recover_and_verify_category_maps()
    # Look up ACH code strictly by NAME from the verified mapping
    ach_code = [k for k, v in inv_pf.items() if v == "ACH"][0]

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_and_build_features("tab_dyn", "temporal", "Small_HI")
    te_prev = float(y_te.mean())
    n_te_total = len(y_te)
    n_te_pos = int(y_te.sum())

    # Load LGBM predictions
    scores_lgbm = np.load(PREDS_DIR / "ablation_tab_dyn_temporal_lightgbm_undersample_seed0_te_scores.npy")
    rng = np.random.RandomState(0)
    tie_breaker = rng.rand(len(scores_lgbm))
    sorted_lgbm_idx = np.lexsort((tie_breaker, scores_lgbm))[::-1]

    # Evaluate a binary selection mask
    def eval_selection(mask: np.ndarray, name: str) -> dict:
        flagged = int(mask.sum())
        tp = int((y_te[mask] == 1).sum())
        prec = (tp / flagged) if flagged > 0 else 0.0
        rec = (tp / n_te_pos) if n_te_pos > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        lift = prec / te_prev if te_prev > 0 else 0.0
        return {
            "Strategy / Model": name,
            "Flagged Alerts (N)": f"{flagged:,}",
            "True Positives (TP)": f"{tp:,}",
            "Precision (%)": f"{prec * 100:.2f}%",
            "Recall (%)": f"{rec * 100:.2f}%",
            "F1 Score": f"{f1:.4f}",
            "Lift over Prevalence": f"{lift:.2f}x",
            "Test PR-AUC": "N/A",
        }

    # 1. Corrected high-risk rule: (Payment Format == ACH) AND (pair_prior_cnt == 0)
    rule_ach = (X_te["payment_format"] == ach_code) & (X_te["pair_prior_cnt"] == 0.0)
    res_rule_ach = eval_selection(rule_ach, "Rule: (Payment Format == ACH) & (pair_prior == 0)")

    # LGBM at same N as ACH rule
    n_ach = int(rule_ach.sum())
    mask_lgbm_ach = np.zeros(n_te_total, dtype=bool)
    mask_lgbm_ach[sorted_lgbm_idx[:n_ach]] = True
    res_lgbm_ach = eval_selection(mask_lgbm_ach, f"LightGBM Tab+Dyn (Top-{n_ach:,} Alerts)")

    # 2. DecisionTree depth 3 on X_tr
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    dt.fit(X_tr, y_tr)
    dt_scores = dt.predict_proba(X_te)[:, 1]
    dt_pr_auc = float(average_precision_score(y_te, dt_scores))
    dt_lift = dt_pr_auc / te_prev

    # DecisionTree optimal F1 point or default 0.5 threshold
    dt_pred = dt.predict(X_te)
    res_dt = eval_selection(dt_pred == 1, "Decision Tree (max_depth=3, Balanced)")
    res_dt["Test PR-AUC"] = f"{dt_pr_auc:.4f}"
    res_dt["Lift over Prevalence"] = f"{dt_lift:.2f}x (PR-AUC Lift)"

    # 3. Full LightGBM model metrics (read dynamically from results.csv seed 0)
    df_res = pd.read_csv(RESULTS_DIR / "results.csv")
    sub_lgbm = df_res[
        (df_res["experiment"] == "ablation_tab_dyn_temporal")
        & (df_res["model_name"] == "lightgbm_undersample")
        & (df_res["seed"] == 0)
    ].iloc[0]

    full_lgbm_pr_auc = float(sub_lgbm["te_pr_auc"])
    full_lgbm_lift = full_lgbm_pr_auc / te_prev
    full_lgbm_prec = float(sub_lgbm["te_precision"])
    full_lgbm_rec = float(sub_lgbm["te_recall"])
    full_lgbm_f1 = float(sub_lgbm["te_f1"])

    res_full_lgbm = {
        "Strategy / Model": "Full LightGBM Model (tab_dyn, Seed 0)",
        "Flagged Alerts (N)": "Continuous (Ranking)",
        "True Positives (TP)": f"{n_te_pos:,} (Total)",
        "Precision (%)": f"{full_lgbm_prec * 100:.2f}% (@ Val Thr)",
        "Recall (%)": f"{full_lgbm_rec * 100:.2f}% (@ Val Thr)",
        "F1 Score": f"{full_lgbm_f1:.4f}",
        "Lift over Prevalence": f"{full_lgbm_lift:.2f}x (PR-AUC Lift)",
        "Test PR-AUC": f"{full_lgbm_pr_auc:.4f}",
    }

    comparison_df = pd.DataFrame([
        res_rule_ach,
        res_lgbm_ach,
        res_dt,
        res_full_lgbm,
    ])

    return comparison_df


def results_tenure_check_table() -> pd.DataFrame:
    """2.g: Tenure check table loaded from results/tenure_check.csv."""
    tenure_path = RESULTS_DIR / "tenure_check.csv"
    if not tenure_path.exists():
        raise FileNotFoundError(f"{tenure_path} not found!")
    df_tenure = pd.read_csv(tenure_path)
    df_tenure["lift"] = df_tenure["test_pr_auc"] / df_tenure["test_prevalence"]
    return df_tenure


def results_shap_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2.h: Reads verified direction and category-code tables from CSV artifacts."""
    dir_path = RESULTS_DIR / "direction_table.csv"
    if not dir_path.exists():
        raise FileNotFoundError(f"{dir_path} not found! Run explain_checks.py first.")
    df_dir = pd.read_csv(dir_path)

    cat_path = RESULTS_DIR / "category_shap_corrected.csv"
    if not cat_path.exists():
        raise FileNotFoundError(f"{cat_path} not found! Run explain_checks.py first.")
    df_cat = pd.read_csv(cat_path)

    return df_dir, df_cat

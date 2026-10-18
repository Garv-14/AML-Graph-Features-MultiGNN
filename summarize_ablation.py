"""
Generate results/summary_ablation.md from results/results.csv.

Feature sets:
  tabular          -> 11 features
  tab_dyn          -> 31 features (11 tabular + 20 dynamic)
  tab_dyn_static   -> 37 features (11 tabular + 20 dynamic + 6 static)

Experiments in results.csv:
  tabular_temporal / tabular_random
  ablation_tab_dyn_temporal / ablation_tab_dyn_random
  ablation_tab_dyn_static_temporal / ablation_tab_dyn_static_random
"""

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_CSV = Path("results/results.csv")
OUT_MD = Path("results/summary_ablation.md")

EXP_MAP = {
    "tabular_temporal":                ("tabular",        "temporal"),
    "tabular_random":                  ("tabular",        "random"),
    "ablation_tab_dyn_temporal":       ("tab_dyn",        "temporal"),
    "ablation_tab_dyn_random":         ("tab_dyn",        "random"),
    "ablation_tab_dyn_static_temporal":("tab_dyn_static", "temporal"),
    "ablation_tab_dyn_static_random":  ("tab_dyn_static", "random"),
}

FEATURE_COUNT = {"tabular": 11, "tab_dyn": 31, "tab_dyn_static": 37}
FEATURE_DESC  = {
    "tabular":        "11 tabular",
    "tab_dyn":        "11 tabular + 20 dynamic graph",
    "tab_dyn_static": "11 tabular + 20 dynamic graph + 6 static graph",
}

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm_undersample",
]

MODEL_LABEL = {
    "logistic_regression":  "Logistic Regression",
    "random_forest":        "Random Forest",
    "xgboost":              "XGBoost",
    "lightgbm_undersample": "LightGBM (undersample)",
}

FS_ORDER = ["tabular", "tab_dyn", "tab_dyn_static"]
FS_LABEL = {
    "tabular":        "tabular",
    "tab_dyn":        "tab+dyn",
    "tab_dyn_static": "tab+dyn+static",
}


def load_and_aggregate(csv_path):
    df = pd.read_csv(csv_path)
    df["feature_set"] = df["experiment"].map(lambda e: EXP_MAP.get(e, (None, None))[0])
    df["split"]       = df["experiment"].map(lambda e: EXP_MAP.get(e, (None, None))[1])
    df = df.dropna(subset=["feature_set"])

    agg = (
        df.groupby(["feature_set", "split", "model_name"])
        .agg(
            pr_auc_mean   =("te_pr_auc",  "mean"),
            pr_auc_std    =("te_pr_auc",  "std"),
            roc_auc_mean  =("te_roc_auc", "mean"),
            roc_auc_std   =("te_roc_auc", "std"),
            f1_mean       =("te_f1",      "mean"),
            f1_std        =("te_f1",      "std"),
            prec_mean     =("te_precision","mean"),
            recall_mean   =("te_recall",  "mean"),
            pak_001_mean  =("te_budget_0.001_precision_at_k", "mean"),
            pak_005_mean  =("te_budget_0.005_precision_at_k", "mean"),
            n_seeds       =("seed",       "count"),
        )
        .reset_index()
    )
    return agg


def fmt(val, std=None):
    if std is not None and not np.isnan(std):
        return f"{val:.4f} +/- {std:.4f}"
    return f"{val:.4f}"


def build_full_table(agg, split):
    sub = agg[agg["split"] == split].copy()
    base_pr = sub[sub["feature_set"] == "tabular"].set_index("model_name")["pr_auc_mean"]

    header = (
        f"{'Feature set':<16} | {'Model':<24} | "
        f"{'PR-AUC (mean+/-std)':>22} | {'ROC-AUC':>14} | {'F1':>14} | "
        f"{'P@0.1%':>8} | {'P@0.5%':>8} | {'Delta PR-AUC':>13}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for fs in FS_ORDER:
        for i, model in enumerate(MODEL_ORDER):
            r = sub[(sub["feature_set"] == fs) & (sub["model_name"] == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            b = base_pr.get(model, float("nan"))
            delta = r["pr_auc_mean"] - b
            sign = "+" if delta >= 0 else ""
            fs_label = FS_LABEL[fs] if i == 0 else ""
            lines.append(
                f"{fs_label:<16} | {MODEL_LABEL.get(model, model):<24} | "
                f"{fmt(r['pr_auc_mean'], r['pr_auc_std']):>22} | "
                f"{fmt(r['roc_auc_mean'], r['roc_auc_std']):>14} | "
                f"{fmt(r['f1_mean'], r['f1_std']):>14} | "
                f"{r['pak_001_mean']*100:>7.1f}% | "
                f"{r['pak_005_mean']*100:>7.1f}% | "
                f"{sign}{delta:>+.4f}"
            )
        lines.append(sep)
    return "\n".join(lines)


def main():
    agg = load_and_aggregate(RESULTS_CSV)

    lines = []
    lines.append("# Step 5 Ablation Summary -- Graph Feature Enrichment")
    lines.append("")
    lines.append("Dataset: Small HI-Small (AML) -- 5 M transactions")
    lines.append("Seeds: 0, 1, 2 (mean +/- std across 3 seeds)")
    lines.append("")
    lines.append("## Feature Sets")
    lines.append("")
    for fs in FS_ORDER:
        lines.append(f"- **{FS_LABEL[fs]}** ({FEATURE_COUNT[fs]} features): {FEATURE_DESC[fs]}")
    lines.append("")
    lines.append("## History Coverage (temporal split)")
    lines.append("")
    lines.append("| Set   | Sender no-history | Receiver no-history |")
    lines.append("|-------|-------------------|---------------------|")
    lines.append("| Train |           13.22%  |             11.44%  |")
    lines.append("| Val   |            0.09%  |              0.02%  |")
    lines.append("| Test  |            0.10%  |              0.02%  |")
    lines.append("")
    lines.append("Val/Test rows have near-zero missing history because they follow the dense training period.")
    lines.append("")

    for split in ["temporal", "random"]:
        lines.append(f"---")
        lines.append(f"## {split.capitalize()} Split")
        lines.append("")
        lines.append("### Full metrics table (test set, mean +/- std over 3 seeds)")
        lines.append("")
        lines.append("```")
        lines.append(build_full_table(agg, split))
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("## Lift Gains vs Tabular Baseline")
    lines.append("")

    for split in ["temporal", "random"]:
        sub = agg[agg["split"] == split]
        base_pr = sub[sub["feature_set"] == "tabular"].set_index("model_name")["pr_auc_mean"]
        dyn_pr  = sub[sub["feature_set"] == "tab_dyn"].set_index("model_name")["pr_auc_mean"]
        stat_pr = sub[sub["feature_set"] == "tab_dyn_static"].set_index("model_name")["pr_auc_mean"]

        lines.append(f"### {split.capitalize()} split (PR-AUC delta vs tabular)")
        lines.append("")
        lines.append("| Model | tab | tab+dyn | Gain | tab+dyn+static | Gain vs tab |")
        lines.append("|-------|-----|---------|------|----------------|-------------|")
        for model in MODEL_ORDER:
            b = base_pr.get(model, float("nan"))
            d = dyn_pr.get(model, float("nan"))
            s = stat_pr.get(model, float("nan"))
            d1 = d - b
            d2 = s - b
            lines.append(
                f"| {MODEL_LABEL.get(model, model)} "
                f"| {b:.4f} | {d:.4f} | {d1:+.4f} "
                f"| {s:.4f} | {d2:+.4f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("- **Dynamic graph features (tab+dyn)** give the largest lift on the temporal split,")
    lines.append("  especially for LightGBM and Random Forest.")
    lines.append("- **Static features (tab+dyn+static)** show mixed results -- PageRank/degree computed")
    lines.append("  only on training edges are noisier, and the gains over tab+dyn are not consistent.")
    lines.append("- **LightGBM (undersample)** is the best single model across all feature sets and splits.")
    lines.append("- **Logistic Regression** benefits less from graph features because it cannot capture")
    lines.append("  non-linear interactions.")
    lines.append("")
    lines.append("### Best model per split")
    lines.append("")
    for split in ["temporal", "random"]:
        sub = agg[agg["split"] == split]
        best = sub.loc[sub["pr_auc_mean"].idxmax()]
        lines.append(
            f"- **{split}**: `{FS_LABEL[best['feature_set']]}` + "
            f"`{MODEL_LABEL.get(best['model_name'], best['model_name'])}`"
            f" -> PR-AUC = **{best['pr_auc_mean']:.4f}** +/- {best['pr_auc_std']:.4f}"
        )
    lines.append("")

    md = "\n".join(lines)
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Written: {OUT_MD}")
    print()
    print(md)


if __name__ == "__main__":
    main()

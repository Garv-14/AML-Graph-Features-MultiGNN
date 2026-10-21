"""
update_readme_results.py
------------------------
Reads results/results.csv, results/tenure_check.csv, and results/final_tables.md,
builds a compact results table, and writes it into README.md between the markers:
    <!-- RESULTS:START -->
    <!-- RESULTS:END -->

Run:
    python update_readme_results.py
"""

import pathlib
import re
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent
RESULTS_CSV = ROOT / "results" / "results.csv"
TENURE_CSV = ROOT / "results" / "tenure_check.csv"
FINAL_TABLES_MD = ROOT / "results" / "final_tables.md"
README = ROOT / "README.md"

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"

# ---------------------------------------------------------------------------
# Models to show and their display names
# ---------------------------------------------------------------------------
TABULAR_MODELS = ["random_forest", "xgboost", "lightgbm_undersample"]
MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm_undersample": "LightGBM (Undersample)",
}

# Experiments to include in the table
# experiment name -> (feature_set_label, split_kind)
EXPERIMENTS = {
    "tabular_temporal":              ("tabular",         "temporal"),
    "tabular_random":                ("tabular",         "random"),
    "ablation_tab_dyn_temporal":     ("tab_dyn",         "temporal"),
    "ablation_tab_dyn_random":       ("tab_dyn",         "random"),
    "ablation_tab_dyn_static_temporal": ("tab_dyn_static", "temporal"),
    "ablation_tab_dyn_static_random":   ("tab_dyn_static", "random"),
}


def compute_lift(pr_auc: float, prevalence: float) -> float:
    if prevalence <= 0:
        return float("nan")
    return pr_auc / prevalence


def build_table(df: pd.DataFrame) -> str:
    """
    Build a compact Markdown table of test PR-AUC lift (mean ± std over seeds)
    for the three headline models across all six experiments.
    """
    rows = []
    for exp_name, (feat_label, split_kind) in EXPERIMENTS.items():
        for model in TABULAR_MODELS:
            mask = (df["experiment"] == exp_name) & (df["model_name"] == model)
            sub = df[mask]
            if sub.empty:
                continue
            # compute lift per row
            lifts = sub.apply(
                lambda r: compute_lift(r["te_pr_auc"], r["te_prevalence"]), axis=1
            )
            mean_lift = lifts.mean()
            std_lift = lifts.std(ddof=1) if len(lifts) > 1 else 0.0
            n_seeds = len(sub)
            rows.append(
                {
                    "Feature set": feat_label,
                    "Split": split_kind,
                    "Model": MODEL_DISPLAY[model],
                    "Seeds": n_seeds,
                    "Test Lift (mean)": mean_lift,
                    "Test Lift (±std)": std_lift,
                }
            )

    result_df = pd.DataFrame(rows)

    # Build pivot-like markdown
    lines = []
    lines.append("")
    lines.append("Test PR-AUC **lift** (= PR-AUC / test prevalence), mean ± std over seeds.")
    lines.append("")

    # Header
    col_header = "| Feature set | Split | Model | Seeds | Lift (mean ± std) |"
    col_sep =    "|:------------|:------|:------|------:|------------------:|"
    lines.append(col_header)
    lines.append(col_sep)

    for _, r in result_df.iterrows():
        lift_str = f"{r['Test Lift (mean)']:.1f} ± {r['Test Lift (±std)']:.1f}"
        lines.append(
            f"| {r['Feature set']} | {r['Split']} | {r['Model']} "
            f"| {r['Seeds']} | {lift_str} |"
        )

    # --- Test prevalence and number of seeds ---
    # Use the first temporal tabular row for temporal prevalence, first random for random
    temporal_rows = df[df["split_kind"] == "temporal"]
    random_rows   = df[df["split_kind"] == "random"]

    temporal_prev = temporal_rows["te_prevalence"].iloc[0] if not temporal_rows.empty else float("nan")
    random_prev   = random_rows["te_prevalence"].iloc[0]   if not random_rows.empty  else float("nan")
    n_seeds_used  = int(df["seed"].nunique())

    lines.append("")
    lines.append(
        f"Test prevalence: temporal split = {temporal_prev:.4%}, "
        f"random split = {random_prev:.4%}. "
        f"All experiments run over {n_seeds_used} seeds (0, 1, 2)."
    )
    lines.append("")

    return "\n".join(lines)


def inject_into_readme(table_md: str) -> None:
    text = README.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        print(
            f"ERROR: Could not find markers '{START_MARKER}' / '{END_MARKER}' in README.md",
            file=sys.stderr,
        )
        sys.exit(1)
    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    replacement = START_MARKER + "\n" + table_md + END_MARKER
    new_text = pattern.sub(replacement, text)
    README.write_text(new_text, encoding="utf-8")
    print("README.md updated between result markers.")


def main() -> None:
    # Read results CSV
    df = pd.read_csv(RESULTS_CSV)
    # Read tenure check (used by README for reference; not injected numerically)
    tenure_df = pd.read_csv(TENURE_CSV)  # noqa: F841 – validated on load, not embedded
    # Verify final_tables.md is readable
    _ = FINAL_TABLES_MD.read_text(encoding="utf-8")

    table_md = build_table(df)
    print("Generated results table:")
    print(table_md)
    inject_into_readme(table_md)


if __name__ == "__main__":
    main()

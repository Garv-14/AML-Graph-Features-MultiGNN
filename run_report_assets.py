"""Plain script executing all report assets for Step 7.

Runs all EDA (Section 1) and Results (Section 2) computations from report_assets.py,
saves all 200 dpi figures to results/figures/ with the 'report_' prefix,
persists results/final_tables.md, and prints all summary tables and metrics.
"""

from pathlib import Path
import time
import pandas as pd
import report_assets as ra

def main():
    t0 = time.time()
    print("=" * 80)
    print("STEP 7: REPORT ASSETS & BENCHMARK VISUALIZATIONS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. EDA ASSETS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. EXPLORATORY DATA ANALYSIS (01_eda.ipynb)")
    print("=" * 80)

    print("\n--- 1.a: Overview Table ---")
    df_overview = ra.eda_overview_table()
    print(df_overview.to_string(index=False))

    print("\n--- 1.b: Daily Laundering Rate & Daily Volume ---")
    df_daily = ra.eda_daily_laundering_plot()
    print("Saved figure: results/figures/report_eda_daily_laundering.png")
    print("Daily Laundering Sample (Days 0-5 = Train, 6-7 = Val, 8-17 = Test):")
    print(df_daily.head(8).to_string(index=False))

    print("\n--- 1.c: Payment Format & Sent Currency Distributions ---")
    df_pf, df_curr = ra.eda_payment_format_table_and_plot()
    print("Saved figure: results/figures/report_eda_payment_format.png")
    print("\nPayment Format Distribution:")
    print(df_pf.to_string(index=False))
    print("\nTop 8 Sent Currencies Distribution:")
    print(df_curr.to_string(index=False))

    print("\n--- 1.d: Amounts Distribution ---")
    df_amt = ra.eda_amounts_table_and_plot()
    print("Saved figure: results/figures/report_eda_amounts.png")
    print(df_amt.to_string(index=False))

    print("\n--- 1.e: Hourly & Day-of-Week Patterns ---")
    df_hour, df_dow = ra.eda_time_patterns_plot()
    print("Saved figure: results/figures/report_eda_hourly_weekly.png")
    print("Hourly Laundering Rate (First 6 Hours):")
    print(df_hour.head(6).to_string(index=False))
    print("\nDay-of-Week Laundering Rate:")
    print(df_dow.to_string(index=False))

    print("\n--- 1.f: Account History & Daily Pair Dynamism ---")
    df_cov, df_pair = ra.eda_account_history()
    print("Saved figure: results/figures/report_eda_pair_prior_cnt_daily.png")
    print("History Coverage Across Splits:")
    print(df_cov.to_string(index=False))

    print("\n--- 1.g: Laundering Rate by Prior Pair History ---")
    df_pp = ra.eda_pair_prior_laundering_rate()
    print(df_pp.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. RESULTS ASSETS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. MODEL BENCHMARK RESULTS (02_results.ipynb)")
    print("=" * 80)

    print("\n--- 2.a: Final Benchmark Tables (results/final_tables.md) ---")
    exp_tables = ra.results_final_tables()
    print("Generated results/final_tables.md for experiments:")
    for exp_name, tbl in exp_tables.items():
        print(f"\nExperiment: {exp_name}")
        print(tbl.to_string(index=False))

    print("\n--- 2.b: Model Lift Across Feature Sets ---")
    df_lift = ra.results_lift_by_feature_set_plot()
    print("Saved figure: results/figures/report_results_lift_by_features.png")
    print(df_lift.to_string(index=False))

    print("\n--- 2.c: Temporal vs. Random Split Lift & Inflation Ratio ---")
    df_comp = ra.results_temporal_vs_random_lift_plot()
    print("Saved figure: results/figures/report_results_lift_inflation.png")
    print(df_comp.to_string(index=False))

    print("\n--- 2.d: Precision-Recall Curves (Temporal Test Set) ---")
    pr_res = ra.results_pr_curves_plot()
    print("Saved figure: results/figures/report_results_pr_curves.png")
    print(f"Verified PR-AUC: Tabular = {pr_res['pr_auc_tabular']:.6f}, Tab+Dyn = {pr_res['pr_auc_tab_dyn']:.6f}")

    print("\n--- 2.e: Alert Budget Curves & Table (Temporal Tab+Dyn) ---")
    df_ab = ra.results_alert_budget_table_and_plot()
    print("Saved figure: results/figures/report_results_alert_budgets.png")
    print(df_ab.to_string(index=False))

    print("\n--- 2.f: Simple Rule Comparison Check ---")
    df_rule = ra.results_simple_rule_check()
    print(df_rule.to_string(index=False))

    print("\n--- 2.g: Tenure Ablation Table ---")
    df_tc = ra.results_tenure_check_table()
    print(df_tc.to_string(index=False))

    print("\n--- 2.h: SHAP Interpretability Summary ---")
    df_dir, df_cat = ra.results_shap_summary()
    print("\nTop 15 Feature Directionality:")
    print(df_dir.to_string(index=False))
    print("\nCategorical Channels Breakdown:")
    print(df_cat.to_string(index=False))

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"All report assets computed and generated in {elapsed:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Builds and executes notebooks/01_eda.ipynb and notebooks/02_results.ipynb."""

from pathlib import Path
import nbformat as nbf
import subprocess
import sys

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def build_eda_notebook() -> Path:
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# Exploratory Data Analysis (IBM AML Small_HI)\n\n"
            "This notebook computes and presents exploratory data analysis tables and figures "
            "for the IBM AML `Small_HI` transaction dataset using helper routines from `report_assets.py`."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "import report_assets as ra\n"
            "from IPython.display import display, Image, Markdown\n"
            "print('Loaded report_assets successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Dataset Overview\n\n"
            "The table below shows the total transaction rows, distinct accounts, total duration in days, "
            "laundering transaction count, and overall dataset laundering prevalence in `formatted_transactions.csv`."
        ),
        nbf.v4.new_code_cell(
            "df_overview = ra.eda_overview_table()\n"
            "display(df_overview)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Daily Laundering Rate and Daily Volume Across Split Partitions\n\n"
            "The figure displays the daily laundering prevalence (%) on the upper panel and total daily transactions "
            "on the lower panel, with dashed vertical lines marking the train (days 0-5), validation (days 6-7), "
            "and test (days 8-17) temporal split boundaries."
        ),
        nbf.v4.new_code_cell(
            "df_daily = ra.eda_daily_laundering_plot()\n"
            "display(Image(filename='../results/figures/report_eda_daily_laundering.png'))\n"
            "display(df_daily.head(6))"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Payment Format and Currency Breakdown\n\n"
            "The tables and figure show the distribution across payment formats and the top 8 sent currencies, "
            "verifying row counts against `HI-Small_Trans.csv`, and comparing the proportion of all positives versus negatives."
        ),
        nbf.v4.new_code_cell(
            "df_pf, df_curr = ra.eda_payment_format_table_and_plot()\n"
            "display(Markdown('### Payment Format Distribution'))\n"
            "display(df_pf)\n"
            "display(Image(filename='../results/figures/report_eda_payment_format.png'))\n"
            "display(Markdown('### Top 8 Sent Currencies'))\n"
            "display(df_curr)"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Transaction Amounts Distribution\n\n"
            "The figure displays density-normalized histograms of log10(Amount Sent) comparing all laundering positives (N=5,177) "
            "against a random sample of 200,000 negatives. The table lists the median and 99th percentile amounts."
        ),
        nbf.v4.new_code_cell(
            "df_amt = ra.eda_amounts_table_and_plot()\n"
            "display(df_amt)\n"
            "display(Image(filename='../results/figures/report_eda_amounts.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Temporal Patterns: Hour of Day and Day of Week\n\n"
            "The figure shows laundering rates (%) across the 24 hours of the day (UTC) on the left panel and across days "
            "of the week (Monday through Sunday) on the right panel."
        ),
        nbf.v4.new_code_cell(
            "df_h, df_dow = ra.eda_time_patterns_plot()\n"
            "display(Image(filename='../results/figures/report_eda_hourly_weekly.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Account History and First-Time Pair Dynamics\n\n"
            "The table reports the proportion of transactions with zero prior sender or receiver history across temporal splits. "
            "The line plot displays the daily percentage of first-time pair transactions (`pair_prior_cnt == 0`) across the stream."
        ),
        nbf.v4.new_code_cell(
            "df_cov, df_dp = ra.eda_account_history()\n"
            "display(df_cov)\n"
            "display(Image(filename='../results/figures/report_eda_pair_prior_cnt_daily.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Laundering Rate by Prior Pair History\n\n"
            "The table compares laundering rates and enrichment ratios between first-time directed pairings (`pair_prior_cnt == 0`) "
            "and recurring pairings (`pair_prior_cnt > 0`) across train, validation, and test splits."
        ),
        nbf.v4.new_code_cell(
            "df_pp = ra.eda_pair_prior_laundering_rate()\n"
            "display(df_pp)"
        ),
    ]

    nb.cells = cells
    nb_path = NOTEBOOKS_DIR / "01_eda.ipynb"
    nbf.write(nb, nb_path)
    print(f"Created {nb_path}")
    return nb_path


def build_results_notebook() -> Path:
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# Model Benchmark Results & Evaluation (IBM AML Small_HI)\n\n"
            "This notebook presents benchmark tables and figures comparing tabular baselines, dynamic graph features, "
            "and static graph features across temporal and random splits, using `report_assets.py`."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "import report_assets as ra\n"
            "from IPython.display import display, Image, Markdown\n"
            "print('Loaded report_assets successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Summary Benchmark Tables across Experiments\n\n"
            "The tables below report mean ± std across seeds for test PR-AUC, lift over prevalence, ROC-AUC, F1, precision, "
            "and recall across all experiments. Plain `lightgbm` is excluded due to single-round early stopping (see Step 4c). "
            "Results are also persisted to `results/final_tables.md`."
        ),
        nbf.v4.new_code_cell(
            "exp_tables = ra.results_final_tables()\n"
            "for exp_name, df_table in exp_tables.items():\n"
            "    display(Markdown(f'### Experiment: `{exp_name}`'))\n"
            "    display(df_table)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Model Lift Across Feature Sets\n\n"
            "The figure shows grouped bar charts of PR-AUC lift across models for tabular, tab+dyn, and tab+dyn+static "
            "feature sets on the temporal split (left) and random split (right), with error bars representing std across seeds."
        ),
        nbf.v4.new_code_cell(
            "df_lift = ra.results_lift_by_feature_set_plot()\n"
            "display(Image(filename='../results/figures/report_results_lift_by_features.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Temporal vs. Random Split Performance and Lift Inflation\n\n"
            "The figure compares temporal split lift versus random split lift on the left panel, and shows the inflation "
            "ratio (random lift / temporal lift) on the right panel with a dashed reference line at 1.0x (no inflation)."
        ),
        nbf.v4.new_code_cell(
            "df_comp = ra.results_temporal_vs_random_lift_plot()\n"
            "display(Image(filename='../results/figures/report_results_lift_inflation.png'))\n"
            "display(df_comp)"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Precision-Recall Curves (Temporal Test Set)\n\n"
            "The figure displays test precision-recall curves for `lightgbm_undersample` seed 0 comparing tabular baseline "
            "versus tab+dyn, alongside the test set prevalence baseline."
        ),
        nbf.v4.new_code_cell(
            "pr_res = ra.results_pr_curves_plot()\n"
            "display(Image(filename='../results/figures/report_results_pr_curves.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Inspection Alert Budgets\n\n"
            "The table and curves report Precision@k and Recall@k at compliance inspection budgets of 0.1% (864), "
            "0.5% (4,320), and 1.0% (8,639) of test rows for random forest, xgboost, and lightgbm_undersample on tab+dyn temporal."
        ),
        nbf.v4.new_code_cell(
            "df_ab = ra.results_alert_budget_table_and_plot()\n"
            "display(df_ab)\n"
            "display(Image(filename='../results/figures/report_results_alert_budgets.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Heuristic Rule Check vs. Machine Learning Models\n\n"
            "The table compares test set performance of simple heuristic rules against `lightgbm_undersample` at matching "
            "alert volumes, a depth-3 Decision Tree, and the full continuous LightGBM model."
        ),
        nbf.v4.new_code_cell(
            "df_rule = ra.results_simple_rule_check()\n"
            "display(df_rule)"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Tenure Ablation Check\n\n"
            "The table reports results from `results/tenure_check.csv`, comparing the full tab+dyn model (31 features) "
            "against the ablated model (23 features) where all 8 cumulative volume counters were removed."
        ),
        nbf.v4.new_code_cell(
            "df_tc = ra.results_tenure_check_table()\n"
            "display(df_tc)"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. SHAP Model Interpretability\n\n"
            "The figures and tables below present global feature importance (top 20 features), beeswarm distribution of impacts "
            "(top 15 features), a representative true positive waterfall plot, and verified empirical direction and category tables."
        ),
        nbf.v4.new_code_cell(
            "display(Markdown('**Figure 8a: Global feature importance (mean |SHAP|) across top 20 features.**'))\n"
            "display(Image(filename='../results/figures/shap_global_bar_top20.png'))\n"
            "display(Markdown('**Figure 8b: Summary beeswarm plot of feature impacts for top 15 features.**'))\n"
            "display(Image(filename='../results/figures/shap_beeswarm_top15.png'))\n"
            "display(Markdown('**Figure 8c: Local waterfall plot for top true positive test prediction (score 0.9998).**'))\n"
            "display(Image(filename='../results/figures/shap_waterfall_tp_1.png'))\n"
            "df_dir, df_cat = ra.results_shap_summary()\n"
            "display(Markdown('**Table 8a: Empirical direction analysis for top 15 features.**'))\n"
            "display(df_dir)\n"
            "display(Markdown('**Table 8b: Category code and channel breakdown.**'))\n"
            "display(df_cat)"
        ),
    ]

    nb.cells = cells
    nb_path = NOTEBOOKS_DIR / "02_results.ipynb"
    nbf.write(nb, nb_path)
    print(f"Created {nb_path}")
    return nb_path


if __name__ == "__main__":
    p1 = build_eda_notebook()
    p2 = build_results_notebook()
    print("Both notebooks created successfully.")

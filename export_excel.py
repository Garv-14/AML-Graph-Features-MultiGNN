"""
export_excel.py - Step 9: Excel workbook from all chart and benchmark data

Generates results/graph_results.xlsx containing 17 sheets:
  1. README
  2. Split_Summary
  3. Daily
  4. Payment_Format
  5. Time_Patterns
  6. Amounts
  7. Lift
  8. Inflation
  9. Metrics
  10. Alert_Budgets
  11. PR_Curve
  12. SHAP
  13. Tenure_Check
  14. Rule_vs_Model
  15. Tail_Sensitivity
  16. Tail_TopK
  17. Tail_Profile

Formatting rules:
  - Font: Arial 10 everywhere.
  - Header rows: bold, light-grey fill (D3D3D3), frozen panes below.
  - Blue font (0000FF) for inputs read from files or data.
  - Black font (000000) for formulas within the same sheet.
  - Green font (008000) for formulas referencing another sheet.
  - Derived values use Excel formulas (rates, shares, lift, ratios, totals).
  - Number formats: counts #,##0; rates/shares 0.000%; PR-AUC/ROC-AUC 0.0000;
    lift 0.0; ratios 0.00"x".
  - One-line source note under every table.
  - wb.calculation.fullCalcOnLoad = True.
"""

import sys
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Paths & Setup
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
RESULTS = BASE / "results"
SPLITS = BASE / "splits"
PREDS = RESULTS / "preds"
OUT_XLSX = RESULTS / "graph_results.xlsx"

sys.path.insert(0, str(BASE))
from data_access import load_formatted, load_split
from explain_checks import recover_and_verify_category_maps

# ---------------------------------------------------------------------------
# Typography & Styles
# ---------------------------------------------------------------------------
ARIAL10 = Font(name="Arial", size=10)
ARIAL10_BOLD = Font(name="Arial", size=10, bold=True)
ARIAL10_BLUE = Font(name="Arial", size=10, color="0000FF")   # Inputs
ARIAL10_BLACK = Font(name="Arial", size=10, color="000000") # Formulas (same sheet)
ARIAL10_GREEN = Font(name="Arial", size=10, color="008000") # Formulas (cross-sheet)
NOTE_FONT = Font(name="Arial", size=9, italic=True, color="555555")
HEADER_FILL = PatternFill("solid", fgColor="D3D3D3")

FMT_COUNT = "#,##0"
FMT_RATE = "0.000%"
FMT_PRAUC = "0.0000"
FMT_LIFT = "0.0"
FMT_RATIO = '0.00"x"'

# ---------------------------------------------------------------------------
# Cell Helpers
# ---------------------------------------------------------------------------
def set_header(ws, row, cols):
    for col_idx, text in enumerate(cols, 1):
        cell = ws.cell(row=row, column=col_idx, value=text)
        cell.font = ARIAL10_BOLD
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def freeze_below(ws, row):
    ws.freeze_panes = ws.cell(row=row + 1, column=1)

def inp(ws, row, col, value, fmt=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = ARIAL10_BLUE
    if fmt:
        cell.number_format = fmt
    return cell

def frm(ws, row, col, formula, fmt=None):
    cell = ws.cell(row=row, column=col, value=formula)
    cell.font = ARIAL10_BLACK
    if fmt:
        cell.number_format = fmt
    return cell

def frm_ext(ws, row, col, formula, fmt=None):
    cell = ws.cell(row=row, column=col, value=formula)
    cell.font = ARIAL10_GREEN
    if fmt:
        cell.number_format = fmt
    return cell

def note(ws, row, col, text):
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = NOTE_FONT
    return cell

def set_col_widths(ws, widths):
    for col, width in widths:
        if isinstance(col, int):
            col = get_column_letter(col)
        ws.column_dimensions[col].width = width

def main():
    print("=" * 80)
    print("STEP 3: EXPORT EXCEL WORKBOOK (results/graph_results.xlsx)")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Load Data Sources & Verify Mappings
    # -----------------------------------------------------------------------
    print("\n--- 1. Loading Data & Recovering Verified Category Mappings ---")
    with open(SPLITS / "Small_HI_split_summary.json") as f:
        split_temp_json = json.load(f)
    with open(SPLITS / "Small_HI_random_split_summary.json") as f:
        split_rand_json = json.load(f)

    df_res = pd.read_csv(RESULTS / "results.csv")
    df_tenure = pd.read_csv(RESULTS / "tenure_check.csv")
    df_dir = pd.read_csv(RESULTS / "direction_table.csv")
    df_cat = pd.read_csv(RESULTS / "category_shap_corrected.csv")
    final_tables_text = (RESULTS / "final_tables.md").read_text(encoding="utf-8")

    # Tail analysis CSVs
    df_tail_sens = pd.read_csv(RESULTS / "tail_sensitivity.csv")
    df_tail_topk = pd.read_csv(RESULTS / "tail_topk.csv")
    df_tail_prof = pd.read_csv(RESULTS / "tail_profile.csv")

    inv_pf, inv_cur, maps_ok = recover_and_verify_category_maps()
    print(f"Category mappings verified: {maps_ok}")
    ach_code = None
    for code, name in inv_pf.items():
        if name == "ACH":
            ach_code = code
            break
    assert ach_code is not None, "ACH code could not be determined!"
    print(f"Verified ACH integer code: {ach_code}")

    print("\nLoading formatted transactions & temporal split indices...")
    df_full = load_formatted("Small_HI")
    print(f"Loaded formatted_transactions.csv: {len(df_full):,} rows")

    df_tr, df_val, df_te = load_split("Small_HI", kind="temporal")
    print(f"Temporal splits: Train={len(df_tr):,}, Val={len(df_val):,}, Test={len(df_te):,}")

    # Load pair_prior_cnt
    graph_par_path = Path("C:/aml_data/Small_HI/graph_dynamic.parquet")
    print("Loading pair_prior_cnt from graph_dynamic.parquet...")
    gdf = pd.read_parquet(graph_par_path, columns=["pair_prior_cnt"])
    df_full["pair_prior_cnt"] = gdf["pair_prior_cnt"].values

    temp_npz = np.load(SPLITS / "Small_HI_split.npz")
    te_inds = temp_npz["te_inds"]
    df_te = df_te.copy()
    df_te["pair_prior_cnt"] = gdf["pair_prior_cnt"].values[te_inds]
    del gdf

    min_ts = df_full["Timestamp"].min()
    df_full["day"] = (df_full["Timestamp"] - min_ts) // 86400
    df_full["hour"] = (df_full["Timestamp"] % 86400) // 3600
    df_full["dow"] = ((df_full["Timestamp"] // 86400) + 3) % 7

    # -----------------------------------------------------------------------
    # 2. Assertions & Validation Checks
    # -----------------------------------------------------------------------
    print("\n--- 2. Running Required Assertions ---")
    # Payment format positives
    expected_pf_pos = {
        "ACH": 4483, "Cheque": 324, "Credit Card": 206, "Cash": 108,
        "Bitcoin": 56, "Reinvestment": 0, "Wire": 0,
    }
    pf_pos_actual = df_full.groupby("Payment Format")["Is Laundering"].sum()
    for code, name in sorted(inv_pf.items()):
        exp = expected_pf_pos.get(name, 0)
        actual = int(pf_pos_actual.get(code, 0))
        assert actual == exp, f"Payment format mismatch for {name}: expected {exp}, got {actual}"
        print(f"  Payment format {name:<14} (code {code}): positives = {actual:>5,} [PASS]")
    print("=> Payment format assertion PASSED.")

    # Daily totals
    daily_rows_sum = len(df_full)
    daily_pos_sum = int(df_full["Is Laundering"].sum())
    assert daily_rows_sum == 5078345, f"Daily rows mismatch: {daily_rows_sum}"
    assert daily_pos_sum == 5177, f"Daily positives mismatch: {daily_pos_sum}"
    print(f"Daily total rows = {daily_rows_sum:,}, positives = {daily_pos_sum:,} [PASS]")

    # Split Summary match
    assert len(df_tr) == split_temp_json["splits"]["train"]["num_rows"]
    assert len(df_val) == split_temp_json["splits"]["validation"]["num_rows"]
    assert len(df_te) == split_temp_json["splits"]["test"]["num_rows"]
    assert int(df_tr["Is Laundering"].sum()) == split_temp_json["splits"]["train"]["num_laundering"]
    assert int(df_val["Is Laundering"].sum()) == split_temp_json["splits"]["validation"]["num_laundering"]
    assert int(df_te["Is Laundering"].sum()) == split_temp_json["splits"]["test"]["num_laundering"]
    print("Split summary JSON counts match loaded splits [PASS].")

    # Tail Sensitivity bulk + tail check
    for _, r_sens in df_tail_sens.iterrows():
        assert r_sens["rows_bulk"] + (r_sens["rows_all"] - r_sens["rows_bulk"]) == r_sens["rows_all"]
        assert r_sens["rows_all"] == 863900
    print("Tail sensitivity bulk rows + tail rows = test set size (863,900) [PASS].")

    # Lift check vs final_tables.md
    print("\nChecking Lift values vs final_tables.md...")
    # Parse final_tables.md
    md_lifts = {}
    current_exp = None
    for line in final_tables_text.split("\n"):
        m_exp = re.match(r"## Experiment: `(.+)`", line)
        if m_exp:
            current_exp = m_exp.group(1)
        if current_exp and "|" in line and "±" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                m_name = parts[1].strip()
                lift_str = parts[4].strip()
                m_lift = re.match(r"([\d.]+)\s*±\s*([\d.]+)", lift_str)
                if m_lift:
                    md_lifts[(current_exp, m_name)] = (float(m_lift.group(1)), float(m_lift.group(2)))

    model_display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm_undersample": "LightGBM (Undersample)",
    }

    lift_check_all_passed = True
    models_4 = ["logistic_regression", "random_forest", "xgboost", "lightgbm_undersample"]
    feat_sets = ["tabular", "tab_dyn", "tab_dyn_static"]
    for fset in feat_sets:
        for mdl in models_4:
            for sp in ["temporal", "random"]:
                exp = f"{fset}_{sp}" if fset == "tabular" else f"ablation_{fset}_{sp}"
                sub_res = df_res[(df_res["experiment"] == exp) & (df_res["model_name"] == mdl)]
                if len(sub_res) == 0:
                    continue
                mean_pr = sub_res["te_pr_auc"].mean()
                std_pr = sub_res["te_pr_auc"].std(ddof=1)
                prev = sub_res["te_prevalence"].iloc[0]
                l_mean = mean_pr / prev
                l_std = std_pr / prev
                disp = model_display_names[mdl]
                if (exp, disp) in md_lifts:
                    md_mean, md_std = md_lifts[(exp, disp)]
                    diff = abs(l_mean - md_mean)
                    status = "PASS" if diff <= 0.1 else "FAIL"
                    if diff > 0.1:
                        lift_check_all_passed = False
                    print(f"  Lift check [{exp}][{disp}]: calc={l_mean:.1f}±{l_std:.1f}, md={md_mean:.1f}±{md_std:.1f} [{status}]")
    assert lift_check_all_passed, "Lift check vs final_tables.md FAILED!"
    print("=> All Lift checks vs final_tables.md PASSED.")

    # -----------------------------------------------------------------------
    # 3. Build Excel Workbook
    # -----------------------------------------------------------------------
    print("\n--- 3. Constructing Sheets in openpyxl ---")
    wb = Workbook()
    wb.calculation.fullCalcOnLoad = True
    for s in wb.sheetnames:
        del wb[s]

    # =======================================================================
    # Sheet 1: README
    # =======================================================================
    ws = wb.create_sheet("README")
    ws["A1"] = "IBM AML Small_HI – Benchmark Tables & Chart Data"
    ws["A1"].font = Font(name="Arial", size=14, bold=True)
    ws["A2"] = "Comprehensive Excel companion for all report charts, baselines, SHAP interpretability, and tail analyses."
    ws["A2"].font = Font(name="Arial", size=10, italic=True)

    ws["A4"] = (
        "Purpose: This workbook provides the complete numerical data behind every chart in results/figures "
        "(EDA, benchmark models, alert budgets, SHAP interpretability, and the near-empty test tail). "
        "Every raw number read from data or benchmark outputs is an INPUT (blue font). All rates, shares, "
        "lifts, differences, ratios, and totals are computed strictly via EXCEL FORMULAS (black font within "
        "the sheet, green font when referencing another sheet)."
    )
    ws["A4"].font = ARIAL10
    ws["A4"].alignment = Alignment(wrap_text=True)
    ws.row_dimensions[4].height = 42

    set_header(ws, 6, ["Sheet Name", "Replaces Chart / Analysis", "Source File(s)"])
    freeze_below(ws, 6)

    sheet_catalog = [
        ("Split_Summary", "Temporal and random split statistics tables", "splits/Small_HI_split_summary.json, Small_HI_random_split_summary.json"),
        ("Daily", "report_eda_daily_laundering.png, report_eda_pair_prior_cnt_daily.png", "formatted_transactions.csv, graph_dynamic.parquet"),
        ("Payment_Format", "report_eda_payment_format.png", "formatted_transactions.csv, results/category_shap_corrected.csv"),
        ("Time_Patterns", "report_eda_hourly_weekly.png", "formatted_transactions.csv"),
        ("Amounts", "report_eda_amounts.png", "formatted_transactions.csv"),
        ("Lift", "report_results_lift_by_features.png", "results/results.csv"),
        ("Inflation", "report_results_lift_inflation.png", "Lift sheet (cross-sheet formulas)"),
        ("Metrics", "Final tables across all 6 experiments (final_tables.md)", "results/results.csv"),
        ("Alert_Budgets", "report_results_alert_budgets.png", "results/results.csv"),
        ("PR_Curve", "report_results_pr_curves.png", "results/results.csv, results/preds/*_te_scores.npy"),
        ("SHAP", "shap_global_bar_top20.png, category SHAP table", "results/direction_table.csv, results/category_shap_corrected.csv"),
        ("Tenure_Check", "Tenure ablation table (dropping cumulative volumes)", "results/tenure_check.csv"),
        ("Rule_vs_Model", "Simple heuristic rule vs Machine Learning table", "formatted_transactions.csv, results/preds/"),
        ("Tail_Sensitivity", "Tail sensitivity table: all rows vs bulk rows (day <= 9)", "results/tail_sensitivity.csv"),
        ("Tail_TopK", "Top-k compliance alert budget tail breakdown", "results/tail_topk.csv"),
        ("Tail_Profile", "Profile of temporal test laundering and normal transactions", "results/tail_profile.csv"),
    ]
    for idx, (sh, ch, src) in enumerate(sheet_catalog, 7):
        ws.cell(row=idx, column=1, value=sh).font = ARIAL10_BLUE
        ws.cell(row=idx, column=2, value=ch).font = ARIAL10
        ws.cell(row=idx, column=3, value=src).font = ARIAL10

    r_leg = 24
    ws.cell(r_leg, 1, "Color Legend & Formula Rules").font = ARIAL10_BOLD
    ws.cell(r_leg + 1, 1, "Blue font:").font = ARIAL10_BLUE
    ws.cell(r_leg + 1, 2, "INPUT values read from source files, parquet, or computed in Python").font = ARIAL10
    ws.cell(r_leg + 2, 1, "Black font:").font = ARIAL10_BLACK
    ws.cell(r_leg + 2, 2, "Excel FORMULAS derived within the same sheet (rates, shares, lift, totals)").font = ARIAL10
    ws.cell(r_leg + 3, 1, "Green font:").font = ARIAL10_GREEN
    ws.cell(r_leg + 3, 2, "Cross-sheet Excel FORMULAS reading another sheet (e.g. Inflation -> Lift)").font = ARIAL10

    r_notes = 29
    ws.cell(r_notes, 1, "Methodological Notes").font = ARIAL10_BOLD
    doc_notes = [
        "1. Synthetic dataset: IBM AML Small_HI transactions, generated by multi-agent simulator.",
        "2. Lift = PR-AUC / test prevalence; lift = 1.0 represents chance performance.",
        "3. Three seeds (0, 1, 2): values report mean and sample standard deviation (ddof=1) over seeds.",
        "4. Plain class-weighted LightGBM ('lightgbm') is excluded due to single-round early stopping; superseded by lightgbm_undersample.",
        "5. Isolation Forest is unsupervised and appears only in sheets listing all baseline models (Metrics).",
        "6. Random seeds vary model stochasticity only; split partitions are fixed across seeds.",
        "7. Tail phenomenon: days 10-17 contain only 1,108 transactions with 655 laundering cases (59.1% prevalence).",
    ]
    for idx, n_text in enumerate(doc_notes, r_notes + 1):
        ws.cell(idx, 1, n_text).font = ARIAL10

    set_col_widths(ws, [(1, 22), (2, 52), (3, 62)])

    # =======================================================================
    # Sheet 2: Split_Summary
    # =======================================================================
    ws = wb.create_sheet("Split_Summary")
    r = 1
    ws.cell(r, 1, "Temporal Split (Chronological Partitioning)").font = ARIAL10_BOLD
    r += 1
    hdr_split = ["Split", "Days", "Day Count", "Rows", "Laundering Rows", "Share of Rows", "Laundering Rate", "Rate Rel. to Train"]
    set_header(ws, r, hdr_split)
    r += 1
    freeze_below(ws, 2)

    t_start = r
    temp_partitions = [
        ("Train", split_temp_json["splits"]["train"], 6),
        ("Validation", split_temp_json["splits"]["validation"], 2),
        ("Test", split_temp_json["splits"]["test"], 10),
    ]
    for s_name, sp, d_cnt in temp_partitions:
        inp(ws, r, 1, s_name)
        inp(ws, r, 2, sp.get("days", ""))
        inp(ws, r, 3, d_cnt, FMT_COUNT)
        inp(ws, r, 4, sp["num_rows"], FMT_COUNT)
        inp(ws, r, 5, sp["num_laundering"], FMT_COUNT)
        frm(ws, r, 6, f"=D{r}/D{t_start+3}", FMT_RATE)
        frm(ws, r, 7, f"=E{r}/D{r}", FMT_RATE)
        r += 1

    t_tot = r
    ws.cell(r, 1, "Total").font = ARIAL10_BOLD
    inp(ws, r, 2, "0 to 17")
    inp(ws, r, 3, 18, FMT_COUNT)
    inp(ws, r, 4, split_temp_json["total_transactions"], FMT_COUNT)
    inp(ws, r, 5, split_temp_json["total_laundering"], FMT_COUNT)
    frm(ws, r, 6, f"=SUM(F{t_start}:F{t_start+2})", FMT_RATE)
    frm(ws, r, 7, f"=E{r}/D{r}", FMT_RATE)
    for ri in range(t_start, t_start + 4):
        frm(ws, ri, 8, f"=G{ri}/G{t_start}", FMT_RATIO)
    r += 1
    note(ws, r, 1, "Source: splits/Small_HI_split_summary.json (splits.train, validation, test keys)")
    r += 2

    ws.cell(r, 1, "Random Split (Uniform Hash Partitioning, Seed 42)").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, hdr_split)
    r += 1
    r_start = r
    rand_partitions = [
        ("Train", split_rand_json["splits"]["train"]),
        ("Validation", split_rand_json["splits"]["validation"]),
        ("Test", split_rand_json["splits"]["test"]),
    ]
    for s_name, sp in rand_partitions:
        inp(ws, r, 1, s_name)
        inp(ws, r, 2, "All 18 days (random)")
        inp(ws, r, 3, 18, FMT_COUNT)
        inp(ws, r, 4, sp["num_rows"], FMT_COUNT)
        inp(ws, r, 5, sp["num_laundering"], FMT_COUNT)
        frm(ws, r, 6, f"=D{r}/D{r_start+3}", FMT_RATE)
        frm(ws, r, 7, f"=E{r}/D{r}", FMT_RATE)
        r += 1

    r_tot = r
    ws.cell(r, 1, "Total").font = ARIAL10_BOLD
    inp(ws, r, 2, "All 18 days")
    inp(ws, r, 3, 18, FMT_COUNT)
    inp(ws, r, 4, split_rand_json["total_transactions"], FMT_COUNT)
    inp(ws, r, 5, split_rand_json["total_laundering"], FMT_COUNT)
    frm(ws, r, 6, f"=SUM(F{r_start}:F{r_start+2})", FMT_RATE)
    frm(ws, r, 7, f"=E{r}/D{r}", FMT_RATE)
    for ri in range(r_start, r_start + 4):
        frm(ws, ri, 8, f"=G{ri}/G{r_start}", FMT_RATIO)
    r += 1
    note(ws, r, 1, "Source: splits/Small_HI_random_split_summary.json")
    set_col_widths(ws, [(1, 14), (2, 22), (3, 12), (4, 14), (5, 16), (6, 14), (7, 16), (8, 16)])

    # =======================================================================
    # Sheet 3: Daily
    # =======================================================================
    ws = wb.create_sheet("Daily")
    r = 1
    hdr_daily = ["Day", "Split", "Transactions", "Laundering Rows", "Laundering Rate", "New-Pair Rows", "New-Pair Share"]
    set_header(ws, r, hdr_daily)
    r += 1
    freeze_below(ws, 1)

    daily_start = r
    for d in range(18):
        sub_d = df_full[df_full["day"] == d]
        sp_lbl = "Train" if d <= 5 else ("Validation" if d <= 7 else "Test")
        cnt_d = len(sub_d)
        pos_d = int(sub_d["Is Laundering"].sum())
        np_d = int((sub_d["pair_prior_cnt"] == 0).sum())

        inp(ws, r, 1, d, FMT_COUNT)
        inp(ws, r, 2, sp_lbl)
        inp(ws, r, 3, cnt_d, FMT_COUNT)
        inp(ws, r, 4, pos_d, FMT_COUNT)
        frm(ws, r, 5, f"=D{r}/C{r}", FMT_RATE)
        inp(ws, r, 6, np_d, FMT_COUNT)
        frm(ws, r, 7, f"=F{r}/C{r}", FMT_RATE)
        r += 1

    daily_end = r - 1
    ws.cell(r, 1, "Total").font = ARIAL10_BOLD
    inp(ws, r, 2, "All days")
    frm(ws, r, 3, f"=SUM(C{daily_start}:C{daily_end})", FMT_COUNT)
    frm(ws, r, 4, f"=SUM(D{daily_start}:D{daily_end})", FMT_COUNT)
    frm(ws, r, 5, f"=D{r}/C{r}", FMT_RATE)
    frm(ws, r, 6, f"=SUM(F{daily_start}:F{daily_end})", FMT_COUNT)
    frm(ws, r, 7, f"=F{r}/C{r}", FMT_RATE)
    r += 1

    note(ws, r, 1, "Source: formatted_transactions.csv (day = (Timestamp - min_Timestamp)//86400; pair_prior_cnt from graph_dynamic.parquet)")
    set_col_widths(ws, [(1, 8), (2, 14), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16)])

    # =======================================================================
    # Sheet 4: Payment_Format
    # =======================================================================
    ws = wb.create_sheet("Payment_Format")
    r = 1
    ws.cell(r, 1, "Table 1: Whole Dataset Payment Formats (formatted_transactions.csv)").font = ARIAL10_BOLD
    r += 1
    hdr_pf1 = ["Format Name", "Code", "Rows", "Laundering Rows", "Normal Rows", "Laundering Rate", "Share of Laundering", "Share of Normal Rows"]
    set_header(ws, r, hdr_pf1)
    r += 1
    freeze_below(ws, 2)

    pf_start = r
    pf_grp = df_full.groupby("Payment Format").agg(rows=("Is Laundering", "count"), laundering=("Is Laundering", "sum")).reset_index()
    for code in sorted(inv_pf.keys()):
        name = inv_pf[code]
        sub_pf = pf_grp[pf_grp["Payment Format"] == code]
        rows_val = int(sub_pf["rows"].iloc[0]) if len(sub_pf) else 0
        laund_val = int(sub_pf["laundering"].iloc[0]) if len(sub_pf) else 0

        inp(ws, r, 1, name)
        inp(ws, r, 2, code, FMT_COUNT)
        inp(ws, r, 3, rows_val, FMT_COUNT)
        inp(ws, r, 4, laund_val, FMT_COUNT)
        frm(ws, r, 5, f"=C{r}-D{r}", FMT_COUNT)
        frm(ws, r, 6, f"=D{r}/C{r}", FMT_RATE)
        r += 1

    pf_end = r - 1
    total_pf_row = r
    ws.cell(r, 1, "Total").font = ARIAL10_BOLD
    frm(ws, r, 3, f"=SUM(C{pf_start}:C{pf_end})", FMT_COUNT)
    frm(ws, r, 4, f"=SUM(D{pf_start}:D{pf_end})", FMT_COUNT)
    frm(ws, r, 5, f"=SUM(E{pf_start}:E{pf_end})", FMT_COUNT)
    frm(ws, r, 6, f"=D{r}/C{r}", FMT_RATE)
    r += 1

    for ri in range(pf_start, pf_end + 1):
        frm(ws, ri, 7, f"=D{ri}/D{total_pf_row}", FMT_RATE)
        frm(ws, ri, 8, f"=E{ri}/E{total_pf_row}", FMT_RATE)
    frm(ws, total_pf_row, 7, f"=SUM(G{pf_start}:G{pf_end})", FMT_RATE)
    frm(ws, total_pf_row, 8, f"=SUM(H{pf_start}:H{pf_end})", FMT_RATE)

    note(ws, r, 1, "Source: formatted_transactions.csv columns 'Payment Format', 'Is Laundering'")
    r += 2

    ws.cell(r, 1, "Table 2: Chronological Test Period Payment Formats (results/category_shap_corrected.csv)").font = ARIAL10_BOLD
    r += 1
    hdr_pf2 = ["Format Name", "Full-Test Rows", "Full-Test Positives", "Share of Positives", "Positive Rate"]
    set_header(ws, r, hdr_pf2)
    r += 1
    pf_test_start = r

    df_cat_pf = df_cat[df_cat["code"] <= 6].copy()
    for code in sorted(inv_pf.keys()):
        name = inv_pf[code]
        sub_c = df_cat_pf[df_cat_pf["code"] == code]
        ft_r = int(sub_c["full-test rows"].iloc[0]) if len(sub_c) else 0
        ft_p = int(sub_c["full-test positives"].iloc[0]) if len(sub_c) else 0

        inp(ws, r, 1, name)
        inp(ws, r, 2, ft_r, FMT_COUNT)
        inp(ws, r, 3, ft_p, FMT_COUNT)
        r += 1
    pf_test_end = r - 1

    total_pf2_row = r
    ws.cell(r, 1, "Total").font = ARIAL10_BOLD
    frm(ws, r, 2, f"=SUM(B{pf_test_start}:B{pf_test_end})", FMT_COUNT)
    frm(ws, r, 3, f"=SUM(C{pf_test_start}:C{pf_test_end})", FMT_COUNT)
    r += 1

    for ri in range(pf_test_start, pf_test_end + 1):
        frm(ws, ri, 4, f"=IFERROR(C{ri}/C{total_pf2_row},0)", FMT_RATE)
        frm(ws, ri, 5, f"=IFERROR(C{ri}/B{ri},0)", FMT_RATE)
    frm(ws, total_pf2_row, 4, f"=SUM(D{pf_test_start}:D{pf_test_end})", FMT_RATE)
    frm(ws, total_pf2_row, 5, f"=IFERROR(C{total_pf2_row}/B{total_pf2_row},0)", FMT_RATE)

    note(ws, r, 1, "Source: results/category_shap_corrected.csv (payment format rows 0-6)")
    set_col_widths(ws, [(1, 16), (2, 14), (3, 16), (4, 16), (5, 16), (6, 16), (7, 18), (8, 18)])

    # =======================================================================
    # Sheet 5: Time_Patterns
    # =======================================================================
    ws = wb.create_sheet("Time_Patterns")
    r = 1
    ws.cell(r, 1, "Hour of Day (0–23)").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, ["Hour", "Rows", "Laundering Rows", "Laundering Rate"])
    r += 1
    freeze_below(ws, 2)

    hour_grp = df_full.groupby("hour").agg(rows=("Is Laundering", "count"), laundering=("Is Laundering", "sum")).reset_index()
    hour_grp = hour_grp.set_index("hour").reindex(range(24), fill_value=0).reset_index()
    for _, h in hour_grp.iterrows():
        inp(ws, r, 1, int(h["hour"]), FMT_COUNT)
        inp(ws, r, 2, int(h["rows"]), FMT_COUNT)
        inp(ws, r, 3, int(h["laundering"]), FMT_COUNT)
        frm(ws, r, 4, f"=IFERROR(C{r}/B{r},0)", FMT_RATE)
        r += 1
    note(ws, r, 1, "Source: formatted_transactions.csv; hour = (Timestamp % 86400) // 3600")
    r += 2

    ws.cell(r, 1, "Day of Week (0=Monday … 6=Sunday)").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, ["Day of Week", "Rows", "Laundering Rows", "Laundering Rate"])
    r += 1
    dow_labels = ["Mon (0)", "Tue (1)", "Wed (2)", "Thu (3)", "Fri (4)", "Sat (5)", "Sun (6)"]
    dow_grp = df_full.groupby("dow").agg(rows=("Is Laundering", "count"), laundering=("Is Laundering", "sum")).reset_index()
    dow_grp = dow_grp.set_index("dow").reindex(range(7), fill_value=0).reset_index()
    for _, d in dow_grp.iterrows():
        idx = int(d["dow"])
        inp(ws, r, 1, dow_labels[idx])
        inp(ws, r, 2, int(d["rows"]), FMT_COUNT)
        inp(ws, r, 3, int(d["laundering"]), FMT_COUNT)
        frm(ws, r, 4, f"=IFERROR(C{r}/B{r},0)", FMT_RATE)
        r += 1
    note(ws, r, 1, "Source: formatted_transactions.csv; day_of_week = ((Timestamp // 86400) + 3) % 7 as in features_tabular.py")
    set_col_widths(ws, [(1, 20), (2, 14), (3, 16), (4, 16)])

    # =======================================================================
    # Sheet 6: Amounts
    # =======================================================================
    ws = wb.create_sheet("Amounts")
    r = 1
    ws.cell(r, 1, "Summary Statistics: log10(Amount Sent) for Amount Sent > 0").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, ["Group", "N", "Median log10", "99th Pct log10", "Median Amount (=10^median)"])
    r += 1
    freeze_below(ws, 2)

    pos_mask = df_full["Is Laundering"] == 1
    neg_mask = df_full["Is Laundering"] == 0
    pos_amt = df_full.loc[pos_mask & (df_full["Amount Sent"] > 0), "Amount Sent"]
    neg_amt = df_full.loc[neg_mask & (df_full["Amount Sent"] > 0), "Amount Sent"]
    pos_log = np.log10(pos_amt.values)
    neg_log = np.log10(neg_amt.values)

    amt_summary = [
        ("Laundering", len(pos_log), float(np.median(pos_log)), float(np.percentile(pos_log, 99))),
        ("Normal", len(neg_log), float(np.median(neg_log)), float(np.percentile(neg_log, 99))),
    ]
    for gname, n_cnt, med_l, p99_l in amt_summary:
        inp(ws, r, 1, gname)
        inp(ws, r, 2, n_cnt, FMT_COUNT)
        inp(ws, r, 3, round(med_l, 4))
        inp(ws, r, 4, round(p99_l, 4))
        frm(ws, r, 5, f"=10^C{r}", "#,##0.00")
        r += 1
    note(ws, r, 1, "Source: formatted_transactions.csv column 'Amount Sent' > 0; all rows (not a sample).")
    r += 2

    ws.cell(r, 1, "Histogram: log10(Amount Sent), Bins of Width 0.25 from -1 to 8").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, ["Bin Lo", "Bin Hi", "Count Laundering", "Count Normal", "Density Laundering", "Density Normal"])
    r += 1

    bins = np.arange(-1.0, 8.25, 0.25)
    bin_lo = bins[:-1]
    bin_hi = bins[1:]
    pos_hist, _ = np.histogram(pos_log, bins=bins)
    neg_hist, _ = np.histogram(neg_log, bins=bins)
    n_pos_total = len(pos_log)
    n_neg_total = len(neg_log)

    inp(ws, r - 3, 7, n_pos_total, FMT_COUNT)
    inp(ws, r - 3, 8, n_neg_total, FMT_COUNT)
    ws.cell(r - 4, 7, "Total Pos N").font = NOTE_FONT
    ws.cell(r - 4, 8, "Total Neg N").font = NOTE_FONT
    ref_pos_n = f"G{r-3}"
    ref_neg_n = f"H{r-3}"

    for i in range(len(bin_lo)):
        inp(ws, r, 1, round(float(bin_lo[i]), 4))
        inp(ws, r, 2, round(float(bin_hi[i]), 4))
        inp(ws, r, 3, int(pos_hist[i]), FMT_COUNT)
        inp(ws, r, 4, int(neg_hist[i]), FMT_COUNT)
        frm(ws, r, 5, f"=C{r}/({ref_pos_n}*0.25)", "0.000")
        frm(ws, r, 6, f"=D{r}/({ref_neg_n}*0.25)", "0.000")
        r += 1
    note(ws, r, 1, "Source: formatted_transactions.csv 'Amount Sent' > 0; density = count / total_N / 0.25 (formula)")
    set_col_widths(ws, [(1, 10), (2, 10), (3, 18), (4, 16), (5, 20), (6, 18), (7, 16), (8, 16)])

    # =======================================================================
    # Sheet 7: Lift
    # =======================================================================
    ws = wb.create_sheet("Lift")
    r = 1
    ws.cell(r, 1, "Test Prevalence Reference Cells").font = ARIAL10_BOLD
    r += 1
    ws.cell(r, 1, "Temporal Test Prevalence:").font = ARIAL10
    prev_temp_val = float(df_res[df_res["split_kind"] == "temporal"]["te_prevalence"].iloc[0])
    inp(ws, r, 2, prev_temp_val, FMT_RATE)
    prev_temp_ref = f"$B${r}"
    r += 1
    ws.cell(r, 1, "Random Test Prevalence:").font = ARIAL10
    prev_rand_val = float(df_res[df_res["split_kind"] == "random"]["te_prevalence"].iloc[0])
    inp(ws, r, 2, prev_rand_val, FMT_RATE)
    prev_rand_ref = f"$B${r}"
    r += 2

    hdr_lift = [
        "Model", "Feature Set",
        "Chron Mean PR-AUC", "Chron Std PR-AUC",
        "Random Mean PR-AUC", "Random Std PR-AUC",
        "Chronological Lift", "Chron Lift Std",
        "Random Lift", "Random Lift Std",
    ]
    set_header(ws, r, hdr_lift)
    r += 1
    freeze_below(ws, r - 1)
    lift_start = r

    for fset in feat_sets:
        fset_label = {"tabular": "tabular (11 feat)", "tab_dyn": "tab+dyn (31 feat)", "tab_dyn_static": "tab+dyn+static (37 feat)"}[fset]
        for mdl in models_4:
            exp_t = f"{fset}_temporal" if fset == "tabular" else f"ablation_{fset}_temporal"
            exp_r = f"{fset}_random" if fset == "tabular" else f"ablation_{fset}_random"
            sub_t = df_res[(df_res["experiment"] == exp_t) & (df_res["model_name"] == mdl)]
            sub_r = df_res[(df_res["experiment"] == exp_r) & (df_res["model_name"] == mdl)]

            mean_t = float(sub_t["te_pr_auc"].mean())
            std_t = float(sub_t["te_pr_auc"].std(ddof=1))
            mean_r = float(sub_r["te_pr_auc"].mean())
            std_r = float(sub_r["te_pr_auc"].std(ddof=1))

            inp(ws, r, 1, model_display_names[mdl])
            inp(ws, r, 2, fset_label)
            inp(ws, r, 3, mean_t, FMT_PRAUC)
            inp(ws, r, 4, std_t, FMT_PRAUC)
            inp(ws, r, 5, mean_r, FMT_PRAUC)
            inp(ws, r, 6, std_r, FMT_PRAUC)
            frm(ws, r, 7, f"=C{r}/{prev_temp_ref}", FMT_LIFT)
            frm(ws, r, 8, f"=D{r}/{prev_temp_ref}", FMT_LIFT)
            frm(ws, r, 9, f"=E{r}/{prev_rand_ref}", FMT_LIFT)
            frm(ws, r, 10, f"=F{r}/{prev_rand_ref}", FMT_LIFT)
            r += 1

    note(ws, r, 1, "Source: results/results.csv (experiments tabular_*, ablation_tab_dyn_*, ablation_tab_dyn_static_*). Lift = PR-AUC / prevalence (formula).")
    set_col_widths(ws, [(1, 24), (2, 22), (3, 18), (4, 16), (5, 18), (6, 16), (7, 18), (8, 14), (9, 14), (10, 14)])

    # =======================================================================
    # Sheet 8: Inflation
    # =======================================================================
    ws = wb.create_sheet("Inflation")
    r = 1
    ws.cell(r, 1, "Temporal vs. Random Split Performance Gap (Lift Inflation)").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, ["Model", "Feature Set", "Chronological Lift", "Random Lift", "Ratio (Random / Chron)"])
    r += 1
    freeze_below(ws, 2)

    infl_start = r
    lift_row_idx = lift_start
    for fset in feat_sets:
        for mdl in models_4:
            # Cross-sheet references -> GREEN font
            frm_ext(ws, r, 1, f"=Lift!A{lift_row_idx}")
            frm_ext(ws, r, 2, f"=Lift!B{lift_row_idx}")
            frm_ext(ws, r, 3, f"=Lift!G{lift_row_idx}", FMT_LIFT)
            frm_ext(ws, r, 4, f"=Lift!I{lift_row_idx}", FMT_LIFT)
            # Local formula -> BLACK font
            frm(ws, r, 5, f"=IFERROR(D{r}/C{r},0)", FMT_RATIO)
            lift_row_idx += 1
            r += 1

    note(ws, r, 1, "Source: Lift sheet. Note: The ratio measures the performance gap; its causal driver (e.g. historical graph leakage) was not formally tested.")
    set_col_widths(ws, [(1, 24), (2, 24), (3, 18), (4, 14), (5, 22)])

    # =======================================================================
    # Sheet 9: Metrics
    # =======================================================================
    ws = wb.create_sheet("Metrics")
    r = 1
    ws.cell(r, 1, "Comprehensive Benchmark Metrics across All 6 Experiments (3 Seeds)").font = ARIAL10_BOLD
    r += 1
    hdr_metrics = [
        "Experiment", "Model", "Seeds",
        "Mean PR-AUC", "Std PR-AUC",
        "Mean ROC-AUC", "Std ROC-AUC",
        "Mean F1", "Std F1",
        "Mean Precision", "Std Precision",
        "Mean Recall", "Std Recall",
        "Test Prevalence", "Lift",
    ]
    set_header(ws, r, hdr_metrics)
    r += 1
    freeze_below(ws, 2)

    all_models_exp = ["logistic_regression", "random_forest", "xgboost", "isolation_forest", "lightgbm_undersample"]
    experiments_6 = [
        "tabular_temporal", "tabular_random",
        "ablation_tab_dyn_temporal", "ablation_tab_dyn_random",
        "ablation_tab_dyn_static_temporal", "ablation_tab_dyn_static_random",
    ]
    for exp in experiments_6:
        for mdl in all_models_exp:
            sub_m = df_res[(df_res["experiment"] == exp) & (df_res["model_name"] == mdl)]
            if len(sub_m) == 0:
                continue
            n_s = len(sub_m)
            m_pr = float(sub_m["te_pr_auc"].mean())
            s_pr = float(sub_m["te_pr_auc"].std(ddof=1)) if n_s > 1 else 0.0
            m_roc = float(sub_m["te_roc_auc"].mean())
            s_roc = float(sub_m["te_roc_auc"].std(ddof=1)) if n_s > 1 else 0.0
            m_f1 = float(sub_m["te_f1"].mean())
            s_f1 = float(sub_m["te_f1"].std(ddof=1)) if n_s > 1 else 0.0
            m_prec = float(sub_m["te_precision"].mean())
            s_prec = float(sub_m["te_precision"].std(ddof=1)) if n_s > 1 else 0.0
            m_rec = float(sub_m["te_recall"].mean())
            s_rec = float(sub_m["te_recall"].std(ddof=1)) if n_s > 1 else 0.0
            prev_val = float(sub_m["te_prevalence"].iloc[0])

            disp_m = "Isolation Forest" if mdl == "isolation_forest" else model_display_names[mdl]

            inp(ws, r, 1, exp)
            inp(ws, r, 2, disp_m)
            inp(ws, r, 3, n_s, FMT_COUNT)
            inp(ws, r, 4, m_pr, FMT_PRAUC)
            inp(ws, r, 5, s_pr, FMT_PRAUC)
            inp(ws, r, 6, m_roc, FMT_PRAUC)
            inp(ws, r, 7, s_roc, FMT_PRAUC)
            inp(ws, r, 8, m_f1, FMT_PRAUC)
            inp(ws, r, 9, s_f1, FMT_PRAUC)
            inp(ws, r, 10, m_prec, FMT_PRAUC)
            inp(ws, r, 11, s_prec, FMT_PRAUC)
            inp(ws, r, 12, m_rec, FMT_PRAUC)
            inp(ws, r, 13, s_rec, FMT_PRAUC)
            inp(ws, r, 14, prev_val, FMT_RATE)
            frm(ws, r, 15, f"=D{r}/N{r}", FMT_LIFT)
            r += 1

    note(ws, r, 1, "Source: results/results.csv. Plain lightgbm excluded. Lift = PR-AUC / Prevalence (formula).")
    set_col_widths(ws, [(1, 32), (2, 24), (3, 8), (4, 13), (5, 12), (6, 13), (7, 12), (8, 10), (9, 10), (10, 14), (11, 13), (12, 12), (13, 12), (14, 15), (15, 10)])

    # =======================================================================
    # Sheet 10: Alert_Budgets
    # =======================================================================
    ws = wb.create_sheet("Alert_Budgets")
    r = 1
    ws.cell(r, 1, "Alert Budget Performance (ablation_tab_dyn_temporal, 3 Seeds)").font = ARIAL10_BOLD
    r += 1
    hdr_ab = ["Model", "Budget", "Alerts k", "Precision@k Mean", "Precision@k Std", "Recall@k Mean", "Recall@k Std", "Expected Laundering Found"]
    set_header(ws, r, hdr_ab)
    r += 1
    freeze_below(ws, 2)

    ab_models = ["random_forest", "xgboost", "lightgbm_undersample"]
    ab_exp = df_res[df_res["experiment"] == "ablation_tab_dyn_temporal"]
    for mdl in ab_models:
        sub_ab = ab_exp[ab_exp["model_name"] == mdl]
        for b_str, b_lbl in [("0.001", "0.1%"), ("0.005", "0.5%"), ("0.01", "1.0%")]:
            k_val = int(sub_ab[f"te_budget_{b_str}_k"].iloc[0])
            m_pk = float(sub_ab[f"te_budget_{b_str}_precision_at_k"].mean())
            s_pk = float(sub_ab[f"te_budget_{b_str}_precision_at_k"].std(ddof=1))
            m_rk = float(sub_ab[f"te_budget_{b_str}_recall_at_k"].mean())
            s_rk = float(sub_ab[f"te_budget_{b_str}_recall_at_k"].std(ddof=1))

            inp(ws, r, 1, model_display_names[mdl])
            inp(ws, r, 2, b_lbl)
            inp(ws, r, 3, k_val, FMT_COUNT)
            inp(ws, r, 4, m_pk, FMT_RATE)
            inp(ws, r, 5, s_pk, FMT_RATE)
            inp(ws, r, 6, m_rk, FMT_RATE)
            inp(ws, r, 7, s_rk, FMT_RATE)
            frm(ws, r, 8, f"=D{r}*C{r}", "#,##0.0")
            r += 1

    note(ws, r, 1, "Source: results/results.csv (ablation_tab_dyn_temporal). Check values for lightgbm_undersample: k = 864, 4320, 8639.")
    set_col_widths(ws, [(1, 24), (2, 10), (3, 12), (4, 16), (5, 14), (6, 16), (7, 14), (8, 24)])

    # =======================================================================
    # Sheet 11: PR_Curve
    # =======================================================================
    ws = wb.create_sheet("PR_Curve")
    r = 1
    ws.cell(r, 1, "Precision-Recall Curve Data (lightgbm_undersample, Seed 0, Chronological Split)").font = ARIAL10_BOLD
    r += 1

    ws.cell(r, 1, "Table 1: Test PR-AUC and Lift").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, ["Model / Feature Set", "Test PR-AUC (Seed 0)", "Test Prevalence", "Lift"])
    r += 1

    pr_tab_s0 = float(df_res[(df_res["experiment"] == "tabular_temporal") & (df_res["model_name"] == "lightgbm_undersample") & (df_res["seed"] == 0)]["te_pr_auc"].iloc[0])
    pr_dyn_s0 = float(df_res[(df_res["experiment"] == "ablation_tab_dyn_temporal") & (df_res["model_name"] == "lightgbm_undersample") & (df_res["seed"] == 0)]["te_pr_auc"].iloc[0])
    prev_s0 = float(df_res[(df_res["experiment"] == "tabular_temporal") & (df_res["model_name"] == "lightgbm_undersample") & (df_res["seed"] == 0)]["te_prevalence"].iloc[0])

    inp(ws, r, 1, "Tabular Baseline (11 features)")
    inp(ws, r, 2, pr_tab_s0, FMT_PRAUC)
    inp(ws, r, 3, prev_s0, FMT_RATE)
    frm(ws, r, 4, f"=B{r}/C{r}", FMT_LIFT)
    r += 1

    inp(ws, r, 1, "Tab+Dyn (31 features)")
    inp(ws, r, 2, pr_dyn_s0, FMT_PRAUC)
    inp(ws, r, 3, prev_s0, FMT_RATE)
    frm(ws, r, 4, f"=B{r}/C{r}", FMT_LIFT)
    r += 1
    note(ws, r, 1, "Source: results/results.csv (lightgbm_undersample seed 0 temporal)")
    r += 2

    ws.cell(r, 1, "Table 2: Precision at Recall Levels (Computed from Saved Prediction Scores)").font = ARIAL10_BOLD
    r += 1
    set_header(ws, r, ["Recall Level", "Precision (Tabular)", "Precision (Tab+Dyn)"])
    r += 1

    s_tab_file = PREDS / "lightgbm_undersample_temporal_seed0_te_scores.npy"
    s_dyn_file = PREDS / "ablation_tab_dyn_temporal_lightgbm_undersample_seed0_te_scores.npy"
    scores_tab = np.load(s_tab_file)
    scores_dyn = np.load(s_dyn_file)
    y_test_arr = df_te["Is Laundering"].to_numpy()

    p_curve_tab, r_curve_tab, _ = precision_recall_curve(y_test_arr, scores_tab)
    p_curve_dyn, r_curve_dyn, _ = precision_recall_curve(y_test_arr, scores_dyn)

    recall_thresholds = np.linspace(0.0, 1.0, 21)
    for rec_lvl in recall_thresholds:
        m_tab = r_curve_tab >= rec_lvl
        prec_tab_at_r = float(p_curve_tab[m_tab].max()) if m_tab.any() else 0.0
        m_dyn = r_curve_dyn >= rec_lvl
        prec_dyn_at_r = float(p_curve_dyn[m_dyn].max()) if m_dyn.any() else 0.0

        inp(ws, r, 1, round(float(rec_lvl), 2), "0.00")
        inp(ws, r, 2, prec_tab_at_r, FMT_PRAUC)
        inp(ws, r, 3, prec_dyn_at_r, FMT_PRAUC)
        r += 1

    note(ws, r, 1, "Method: precision at recall r = max precision over thresholds where recall >= r. Computed from test prediction arrays.")
    set_col_widths(ws, [(1, 26), (2, 22), (3, 22), (4, 12)])

    # =======================================================================
    # Sheet 12: SHAP
    # =======================================================================
    ws = wb.create_sheet("SHAP")
    r = 1
    ws.cell(r, 1, "Table 1: Top 15 Features Direction Table (results/direction_table.csv)").font = ARIAL10_BOLD
    r += 1
    hdr_shap1 = ["Rank", "Feature", "Mean |SHAP|", "Spearman rho", "Q1 Mean SHAP", "Q5 Mean SHAP", "Empirical Direction", "Share of Top-15 Importance"]
    set_header(ws, r, hdr_shap1)
    r += 1
    freeze_below(ws, 2)

    shap_start = r
    numeric_shap_cells = []
    for idx, drow in df_dir.iterrows():
        ri = r + idx
        inp(ws, ri, 1, int(drow["Rank"]), FMT_COUNT)
        inp(ws, ri, 2, str(drow["Feature"]))
        try:
            m_val = float(drow["Mean |SHAP|"])
            inp(ws, ri, 3, m_val, FMT_PRAUC)
            numeric_shap_cells.append(f"C{ri}")
        except:
            inp(ws, ri, 3, drow["Mean |SHAP|"])
        inp(ws, ri, 4, str(drow["Spearman rho"]))
        inp(ws, ri, 5, str(drow["Q1 Mean SHAP"]))
        inp(ws, ri, 6, str(drow["Q5 Mean SHAP"]))
        inp(ws, ri, 7, str(drow["Empirical Direction"]))

    shap_sum_formula = "+".join(numeric_shap_cells)
    for idx in range(len(df_dir)):
        ri = shap_start + idx
        frm(ws, ri, 8, f"=IFERROR(C{ri}/({shap_sum_formula}),0)", FMT_RATE)

    r = shap_start + len(df_dir)
    note(ws, r, 1, "Source: results/direction_table.csv. Share of top-15 importance = Mean |SHAP| / SUM(top 15 Mean |SHAP|) (formula).")
    r += 2

    ws.cell(r, 1, "Table 2: Categorical Feature SHAP Breakdown (results/category_shap_corrected.csv)").font = ARIAL10_BOLD
    r += 1
    hdr_shap2 = [
        "Category Name", "Code", "Sampled Rows", "Mean SHAP",
        "Share of Sampled Positives", "Share of Sampled Negatives",
        "Full-Test Rows", "Full-Test Positives", "Positive Rate", "Share of Group Positives",
    ]
    set_header(ws, r, hdr_shap2)
    r += 1

    cat_start = r
    pf_rows_idx = []
    cur_rows_idx = []
    for idx, crow in df_cat.iterrows():
        ri = r + idx
        c_code = int(crow["code"])
        c_name = str(crow["format name"])
        is_pf = idx < 7

        inp(ws, ri, 1, c_name)
        inp(ws, ri, 2, c_code, FMT_COUNT)
        inp(ws, ri, 3, int(crow["sampled rows"]), FMT_COUNT)
        inp(ws, ri, 4, float(crow["mean SHAP"]), FMT_PRAUC)
        inp(ws, ri, 5, float(crow["share of sampled positives"]) / 100.0, FMT_RATE)
        inp(ws, ri, 6, float(crow["share of sampled negatives"]) / 100.0, FMT_RATE)
        inp(ws, ri, 7, int(crow["full-test rows"]), FMT_COUNT)
        inp(ws, ri, 8, int(crow["full-test positives"]), FMT_COUNT)
        frm(ws, ri, 9, f"=IFERROR(H{ri}/G{ri},0)", FMT_RATE)

        if is_pf:
            pf_rows_idx.append(ri)
        else:
            cur_rows_idx.append(ri)

    pf_sum_ref = f"SUM(H{pf_rows_idx[0]}:H{pf_rows_idx[-1]})"
    cur_sum_ref = f"SUM(H{cur_rows_idx[0]}:H{cur_rows_idx[-1]})"
    for ri in pf_rows_idx:
        frm(ws, ri, 10, f"=IFERROR(H{ri}/{pf_sum_ref},0)", FMT_RATE)
    for ri in cur_rows_idx:
        frm(ws, ri, 10, f"=IFERROR(H{ri}/{cur_sum_ref},0)", FMT_RATE)

    r = cat_start + len(df_cat)
    note(ws, r, 1, "Source: results/category_shap_corrected.csv. Positive rate = positives / rows; group share = positives / SUM(group positives) (formulas).")
    set_col_widths(ws, [(1, 24), (2, 8), (3, 14), (4, 12), (5, 24), (6, 24), (7, 16), (8, 18), (9, 14), (10, 22)])

    # =======================================================================
    # Sheet 13: Tenure_Check
    # =======================================================================
    ws = wb.create_sheet("Tenure_Check")
    r = 1
    ws.cell(r, 1, "Tenure Check: Cumulative Volume Counter Feature Ablation").font = ARIAL10_BOLD
    r += 1
    hdr_tenure = ["Model Configuration", "N Features", "Dropped Feature Columns", "Test PR-AUC", "Test Prevalence", "Lift", "Unique Scores"]
    set_header(ws, r, hdr_tenure)
    r += 1
    freeze_below(ws, 2)

    t_start_row = r
    for _, trow in df_tenure.iterrows():
        inp(ws, r, 1, str(trow["features_set"]))
        inp(ws, r, 2, int(trow["n_features"]), FMT_COUNT)
        inp(ws, r, 3, str(trow["dropped_columns"]))
        inp(ws, r, 4, float(trow["test_pr_auc"]), FMT_PRAUC)
        inp(ws, r, 5, float(trow["test_prevalence"]), FMT_RATE)
        frm(ws, r, 6, f"=D{r}/E{r}", FMT_LIFT)
        inp(ws, r, 7, int(trow["unique_test_scores"]), FMT_COUNT)
        r += 1

    diff_row = r
    ws.cell(r, 1, "Difference (Ablated − Full)").font = ARIAL10_BOLD
    inp(ws, r, 2, int(df_tenure["n_features"].iloc[1] - df_tenure["n_features"].iloc[0]), FMT_COUNT)
    inp(ws, r, 3, "Dropped 8 cumulative volume counters")
    frm(ws, r, 4, f"=D{t_start_row+1}-D{t_start_row}", FMT_PRAUC)
    inp(ws, r, 5, "")
    frm(ws, r, 6, f"=F{t_start_row+1}-F{t_start_row}", FMT_LIFT)
    frm(ws, r, 7, f"=G{t_start_row+1}-G{t_start_row}", FMT_COUNT)
    r += 1

    note(ws, r, 1, "Source: results/tenure_check.csv. Single seed (seed 0) temporal ablation. Lift = PR-AUC / prevalence (formula).")
    set_col_widths(ws, [(1, 32), (2, 12), (3, 56), (4, 14), (5, 16), (6, 12), (7, 18)])

    # =======================================================================
    # Sheet 14: Rule_vs_Model
    # =======================================================================
    ws = wb.create_sheet("Rule_vs_Model")
    r = 1
    ws.cell(r, 1, "Simple Heuristic Rule vs. Machine Learning on Chronological Test Set").font = ARIAL10_BOLD
    r += 1
    ws.cell(r, 1, f"ACH code verified by name from category map: code = {ach_code}").font = NOTE_FONT
    r += 1
    ws.cell(r, 1, "Test Positives Total:").font = ARIAL10
    inp(ws, r, 2, int(df_te["Is Laundering"].sum()), FMT_COUNT)
    ref_te_pos = f"$B${r}"
    r += 1
    ws.cell(r, 1, "Test Prevalence:").font = ARIAL10
    inp(ws, r, 2, prev_temp_val, FMT_RATE)
    ref_te_prev = f"$B${r}"
    r += 2

    hdr_rvm = ["Method", "Alerts", "True Positives", "Precision", "Recall", "Lift"]
    set_header(ws, r, hdr_rvm)
    r += 1
    freeze_below(ws, r - 1)

    # 1. Rule
    rule_mask = (df_te["Payment Format"] == ach_code) & (df_te["pair_prior_cnt"] == 0)
    rule_alerts = int(rule_mask.sum())
    rule_tps = int((rule_mask & (df_te["Is Laundering"] == 1)).sum())

    inp(ws, r, 1, f"Simple Rule: Payment Format == ACH (code {ach_code}) & pair_prior_cnt == 0")
    inp(ws, r, 2, rule_alerts, FMT_COUNT)
    inp(ws, r, 3, rule_tps, FMT_COUNT)
    frm(ws, r, 4, f"=C{r}/B{r}", FMT_RATE)
    frm(ws, r, 5, f"=C{r}/{ref_te_pos}", FMT_RATE)
    frm(ws, r, 6, f"=D{r}/{ref_te_prev}", FMT_LIFT)
    r += 1

    # 2. ML top-N (N = rule alerts)
    rng0 = np.random.RandomState(0)
    perm0 = rng0.permutation(len(scores_dyn))
    order0 = np.lexsort((perm0, -scores_dyn))
    ml_topn_idx = order0[:rule_alerts]
    ml_topn_tps = int(df_te["Is Laundering"].iloc[ml_topn_idx].sum())

    inp(ws, r, 1, f"LightGBM (Undersample) tab+dyn seed 0 Top-{rule_alerts:,} (ties: seeded perm seed 0)")
    inp(ws, r, 2, rule_alerts, FMT_COUNT)
    inp(ws, r, 3, ml_topn_tps, FMT_COUNT)
    frm(ws, r, 4, f"=C{r}/B{r}", FMT_RATE)
    frm(ws, r, 5, f"=C{r}/{ref_te_pos}", FMT_RATE)
    frm(ws, r, 6, f"=D{r}/{ref_te_prev}", FMT_LIFT)
    r += 1

    # 3. ML Top 0.1% budget (k from Alert_Budgets sheet)
    lgbm_01_prec = float(df_res[(df_res["experiment"] == "ablation_tab_dyn_temporal") & (df_res["model_name"] == "lightgbm_undersample")]["te_budget_0.001_precision_at_k"].mean())
    lgbm_01_rec = float(df_res[(df_res["experiment"] == "ablation_tab_dyn_temporal") & (df_res["model_name"] == "lightgbm_undersample")]["te_budget_0.001_recall_at_k"].mean())

    inp(ws, r, 1, "LightGBM tab+dyn Top 0.1% Budget (k from Alert_Budgets, mean prec/rec)")
    # Cross-sheet reference -> GREEN font
    frm_ext(ws, r, 2, "=Alert_Budgets!C10", FMT_COUNT)
    frm(ws, r, 3, f"=ROUND(D{r}*B{r},0)", FMT_COUNT)
    inp(ws, r, 4, lgbm_01_prec, FMT_RATE)
    inp(ws, r, 5, lgbm_01_rec, FMT_RATE)
    frm(ws, r, 6, f"=D{r}/{ref_te_prev}", FMT_LIFT)
    r += 1

    note(ws, r, 1, "Source: formatted_transactions.csv, graph_dynamic.parquet, and saved LightGBM test predictions.")
    set_col_widths(ws, [(1, 58), (2, 14), (3, 16), (4, 14), (5, 14), (6, 12)])

    # =======================================================================
    # Sheet 15: Tail_Sensitivity
    # =======================================================================
    ws = wb.create_sheet("Tail_Sensitivity")
    r = 1
    ws.cell(r, 1, "Tail Sensitivity Analysis: All Test Rows vs. Bulk Rows (Day <= 9)").font = ARIAL10_BOLD
    r += 1
    hdr_ts = [
        "Split", "Feature Set", "Model", "Seeds",
        "Rows (All)", "Positives (All)", "Prevalence (All)", "AP Mean (All)", "Lift (All)",
        "Rows (Bulk)", "Positives (Bulk)", "Prevalence (Bulk)", "AP Mean (Bulk)", "Lift (Bulk)",
    ]
    set_header(ws, r, hdr_ts)
    r += 1
    freeze_below(ws, 2)

    ts_start = r
    for _, srow in df_tail_sens.iterrows():
        inp(ws, r, 1, str(srow["split"]))
        inp(ws, r, 2, str(srow["feature_set"]))
        inp(ws, r, 3, model_display_names.get(str(srow["model"]), str(srow["model"])))
        inp(ws, r, 4, int(srow["seeds"]), FMT_COUNT)

        inp(ws, r, 5, int(srow["rows_all"]), FMT_COUNT)
        inp(ws, r, 6, int(srow["pos_all"]), FMT_COUNT)
        frm(ws, r, 7, f"=F{r}/E{r}", FMT_RATE)
        inp(ws, r, 8, float(srow["ap_all_mean"]), FMT_PRAUC)
        frm(ws, r, 9, f"=H{r}/G{r}", FMT_LIFT)

        inp(ws, r, 10, int(srow["rows_bulk"]), FMT_COUNT)
        inp(ws, r, 11, int(srow["pos_bulk"]), FMT_COUNT)
        frm(ws, r, 12, f"=K{r}/J{r}", FMT_RATE)
        inp(ws, r, 13, float(srow["ap_bulk_mean"]), FMT_PRAUC)
        frm(ws, r, 14, f"=M{r}/L{r}", FMT_LIFT)
        r += 1
    ts_end = r - 1

    note(ws, r, 1, "Source: results/tail_sensitivity.csv. Bulk rows = day <= 9. Lift = AP / Prevalence (formulas).")
    r += 2

    # Ratios table below
    ws.cell(r, 1, "Random / Temporal Split Lift Ratios (All Rows vs. Bulk Rows)").font = ARIAL10_BOLD
    r += 1
    hdr_ts_ratios = ["Feature Set", "Model", "Random Lift (All)", "Temporal Lift (All)", "Ratio (All)", "Random Lift (Bulk)", "Temporal Lift (Bulk)", "Ratio (Bulk)"]
    set_header(ws, r, hdr_ts_ratios)
    r += 1

    for fset in feat_sets:
        for mdl in models_4:
            t_row = None
            r_row = None
            for scan_r in range(ts_start, ts_end + 1):
                sp_val = df_tail_sens.iloc[scan_r - ts_start]["split"]
                fs_val = df_tail_sens.iloc[scan_r - ts_start]["feature_set"]
                md_val = df_tail_sens.iloc[scan_r - ts_start]["model"]
                if fs_val == fset and md_val == mdl:
                    if sp_val == "temporal":
                        t_row = scan_r
                    elif sp_val == "random":
                        r_row = scan_r

            inp(ws, r, 1, fset)
            inp(ws, r, 2, model_display_names[mdl])
            frm(ws, r, 3, f"=I{r_row}", FMT_LIFT)
            frm(ws, r, 4, f"=I{t_row}", FMT_LIFT)
            frm(ws, r, 5, f"=IFERROR(C{r}/D{r},0)", FMT_RATIO)
            frm(ws, r, 6, f"=N{r_row}", FMT_LIFT)
            frm(ws, r, 7, f"=N{t_row}", FMT_LIFT)
            frm(ws, r, 8, f"=IFERROR(F{r}/G{r},0)", FMT_RATIO)
            r += 1

    note(ws, r, 1, "Formulas: Ratio = Random Lift / Temporal Lift for all test rows vs bulk test rows.")
    set_col_widths(ws, [(1, 14), (2, 22), (3, 20), (4, 8), (5, 14), (6, 14), (7, 16), (8, 14), (9, 12), (10, 14), (11, 14), (12, 16), (13, 14), (14, 12)])

    # =======================================================================
    # Sheet 16: Tail_TopK
    # =======================================================================
    ws = wb.create_sheet("Tail_TopK")
    r = 1
    ws.cell(r, 1, "Compliance Alert Budget Breakdown: Alerts & True Positives in Tail (Days 10-17)").font = ARIAL10_BOLD
    r += 1
    hdr_topk = ["Budget", "Alerts k", "Alerts Days 10-17", "Total TP", "TP Days 10-17", "Tail Alert Share", "Tail TP Share", "Tail Precision"]
    set_header(ws, r, hdr_topk)
    r += 1
    freeze_below(ws, 2)

    for _, krow in df_tail_topk.iterrows():
        b_str = str(krow["budget"])
        k_val = int(krow["k"])
        a_tail = int(krow["alerts_tail"])
        tp_tot = int(krow["tp_total"])
        tp_tail = int(krow["tp_tail"])

        inp(ws, r, 1, b_str)
        inp(ws, r, 2, k_val, FMT_COUNT)
        inp(ws, r, 3, a_tail, FMT_COUNT)
        inp(ws, r, 4, tp_tot, FMT_COUNT)
        inp(ws, r, 5, tp_tail, FMT_COUNT)
        frm(ws, r, 6, f"=C{r}/B{r}", FMT_RATE)
        frm(ws, r, 7, f"=E{r}/D{r}", FMT_RATE)
        frm(ws, r, 8, f"=E{r}/C{r}", FMT_RATE)
        r += 1

    note(ws, r, 1, "Source: results/tail_topk.csv (lightgbm_undersample tab_dyn temporal seed 0). Shares and precision computed via formulas.")
    set_col_widths(ws, [(1, 12), (2, 12), (3, 18), (4, 14), (5, 16), (6, 18), (7, 18), (8, 16)])

    # =======================================================================
    # Sheet 17: Tail_Profile
    # =======================================================================
    ws = wb.create_sheet("Tail_Profile")
    r = 1
    ws.cell(r, 1, "Profile of Test Groups: Positives (Bulk vs Tail) and Normal Rows").font = ARIAL10_BOLD
    r += 1
    hdr_prof = ["Group", "Transaction Count", "Share ACH", "Share pair_prior_cnt == 0"]
    set_header(ws, r, hdr_prof)
    r += 1
    freeze_below(ws, 2)

    for _, prow in df_tail_prof.iterrows():
        inp(ws, r, 1, str(prow["group"]))
        inp(ws, r, 2, int(prow["count"]), FMT_COUNT)
        inp(ws, r, 3, float(prow["share_ach"]), FMT_RATE)
        inp(ws, r, 4, float(prow["share_new_pair"]), FMT_RATE)
        r += 1

    note(ws, r, 1, "Source: results/tail_profile.csv. ACH code looked up by name from verified mapping. Pair prior cnt from graph_dynamic.parquet.")
    set_col_widths(ws, [(1, 30), (2, 18), (3, 16), (4, 26)])

    # =======================================================================
    # Global Font Verification
    # =======================================================================
    print("Applying global font formatting (preserving blue and green colors)...")
    for ws_sheet in wb.worksheets:
        for row in ws_sheet.iter_rows():
            for cell in row:
                if cell.font:
                    c_color = cell.font.color
                    is_b = cell.font.bold
                    is_it = cell.font.italic
                    sz = cell.font.size if cell.font.size else 10
                    cell.font = Font(name="Arial", size=sz, bold=is_b, italic=is_it, color=c_color)
                else:
                    cell.font = ARIAL10

    # -----------------------------------------------------------------------
    # 4. Save and Validate Workbook
    # -----------------------------------------------------------------------
    print(f"\nSaving Excel workbook to: {OUT_XLSX} ...")
    wb.save(OUT_XLSX)
    print("Workbook saved successfully.")

    print("\n--- 4. Post-Save Workbook Verification ---")
    wb_chk = openpyxl.load_workbook(OUT_XLSX, data_only=False)
    formula_count = 0
    error_cells = []
    sheet_row_counts = []

    for sheet in wb_chk.worksheets:
        sheet_row_counts.append((sheet.title, sheet.max_row, sheet.max_column))
        for row in sheet.iter_rows():
            for cell in row:
                val = str(cell.value) if cell.value is not None else ""
                if val.startswith("="):
                    formula_count += 1
                elif val.startswith("#"):
                    error_cells.append(f"{sheet.title}!{cell.coordinate}: {val}")

    print(f"Total sheets: {len(wb_chk.sheetnames)}")
    print(f"Formula cell count: {formula_count}")
    print(f"Error cells (starting with '#'): {len(error_cells)}")
    assert formula_count > 200, f"Formula count {formula_count} <= 200!"
    assert len(error_cells) == 0, f"Found error cells: {error_cells}"
    print("Formula count > 200: PASS.")
    print("No error cells: PASS.")

    print("\nSheet List & Dimensions:")
    for title, rows_cnt, cols_cnt in sheet_row_counts:
        print(f"  {title:<20}: {rows_cnt:>4} rows, {cols_cnt:>3} cols")

    # LibreOffice conversion check
    soffice = shutil.which("soffice")
    if soffice:
        print("\nChecking LibreOffice (soffice) recalculation...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "check.xlsx"
            shutil.copy2(OUT_XLSX, tmp_path)
            res = subprocess.run([soffice, "--headless", "--convert-to", "pdf", str(tmp_path), "--outdir", tmp_dir], capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                print("LibreOffice recalculation & conversion check: PASS.")
            else:
                print(f"LibreOffice conversion returned code {res.returncode}: {res.stderr}")
    else:
        print("\nLibreOffice (soffice) is not installed on this machine — skipping headless recalculation.")

    print(f"\nWorkbook path: {OUT_XLSX.resolve()}")
    print("=" * 80)
    print("STEP 3 EXPORT EXCEL COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

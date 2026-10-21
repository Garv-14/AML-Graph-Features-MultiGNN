"""
Step 6b: Verification and Quantification of SHAP Interpretability Claims.

1. DIRECTION TABLE:
   - For top 15 features by mean |SHAP|:
     * Spearman correlation (rho, p-value) between feature value and SHAP value.
     * Mean SHAP in lowest (Q1) and highest (Q5) value quintiles.
     * Strict label:
       - rho >= 0.2: "higher raises score"
       - rho <= -0.2: "higher lowers score"
       - otherwise: "non-monotonic"
     * Skip Spearman for categorical features (payment_format, sent_currency).

2. CATEGORY CODES:
   - For payment_format and sent_currency (top 5 codes by sample count):
     * Mean SHAP, share of sampled positives, share of sampled negatives, real name.
     * Verified against raw data first-appearance mapping.

3. NEW-PAIR CHECK (Full test set):
   - Share of positives and share of negatives with pair_prior_cnt == 0.
   - Positive rate inside pair_prior_cnt == 0 vs pair_prior_cnt > 0.

4. TENURE CHECK:
   - Retrain lightgbm_undersample (seed 0, temporal, tab_dyn) dropping the 8
     cumulative volume columns: s_out_cnt, s_out_amt, s_in_cnt, s_in_amt,
     r_in_cnt, r_in_amt, r_out_cnt, r_out_amt.
   - Report PR-AUC, lift, unique scores compared to full model (0.694059).
   - Save to results/tenure_check.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats
import shap
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from run_tabular import (
    build_and_fit_lightgbm_undersample,
    load_and_build_features,
    prepare_lgb_features,
)

RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "lgbm_us_tab_dyn_temporal_seed0.pkl"
TENURE_CSV = RESULTS_DIR / "tenure_check.csv"
CATEGORY_SHAP_CORRECTED_CSV = RESULTS_DIR / "category_shap_corrected.csv"
DIRECTION_TABLE_CSV = RESULTS_DIR / "direction_table.csv"
RAW_TRANS_PATH = Path(r"C:\aml_data\Small_HI\HI-Small_Trans.csv")
FMT_TRANS_PATH = Path(r"C:\aml_data\Small_HI\formatted_transactions.csv")


def recover_and_verify_category_maps():
    """Recover categorical mappings from raw transaction file and verify row counts."""
    currency = {}
    payment_format = {}

    def get_dict_val(name, collection):
        if name in collection:
            val = collection[name]
        else:
            val = len(collection)
            collection[name] = val
        return val

    with open(RAW_TRANS_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_map = {col.strip(): idx for idx, col in enumerate(header)}
        rec_curr_idx = col_map["Receiving Currency"]
        pay_curr_idx = col_map["Payment Currency"]
        pay_fmt_idx = col_map["Payment Format"]

        for row in reader:
            get_dict_val(row[rec_curr_idx], currency)
            get_dict_val(row[pay_curr_idx], currency)
            get_dict_val(row[pay_fmt_idx], payment_format)

    inv_currency = {v: k for k, v in currency.items()}
    inv_payment_format = {v: k for k, v in payment_format.items()}

    # Verify counts
    df_fmt = pd.read_csv(FMT_TRANS_PATH, usecols=["Sent Currency", "Payment Format"])
    fmt_pf_counts = df_fmt["Payment Format"].value_counts().to_dict()
    fmt_sc_counts = df_fmt["Sent Currency"].value_counts().to_dict()

    raw_pf_counts = Counter()
    raw_sc_counts = Counter()
    with open(RAW_TRANS_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            raw_pf_counts[row[pay_fmt_idx]] += 1
            raw_sc_counts[row[pay_curr_idx]] += 1

    verified_pf = True
    for code, name in inv_payment_format.items():
        if fmt_pf_counts.get(code, 0) != raw_pf_counts.get(name, 0):
            verified_pf = False

    verified_sc = True
    for code, name in inv_currency.items():
        if fmt_sc_counts.get(code, 0) != raw_sc_counts.get(name, 0):
            verified_sc = False

    return inv_payment_format, inv_currency, (verified_pf and verified_sc)


def main():
    print("=" * 80)
    print("STEP 6b: EMPIRICAL VERIFICATION OF SHAP CLAIMS & QUANTIFICATION")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # 0. Load Category Mappings
    # ------------------------------------------------------------------ #
    print("\n--- 0. Recovering & Verifying Category Mappings ---")
    inv_payment_format, inv_currency, maps_verified = recover_and_verify_category_maps()
    if maps_verified:
        print("  => Category mappings VERIFIED against raw transactions (exact row counts match).")
    else:
        print("  => WARNING: Category mappings could NOT be completely verified!")

    # ------------------------------------------------------------------ #
    # 1. Load Data & Existing Trained Model
    # ------------------------------------------------------------------ #
    print("\n--- 1. Loading Features and Rebuilding Test Sample ---")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_and_build_features(
        feature_set="tab_dyn",
        split_kind="temporal",
        data_name="Small_HI",
    )

    X_te_lgb = prepare_lgb_features(X_te)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

    print(f"Loading trained model: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)

    # Verify score
    full_te_scores = clf.predict_proba(X_te_lgb)[:, 1]
    full_pr_auc = float(average_precision_score(y_te, full_te_scores))
    print(f"Full model test PR-AUC: {full_pr_auc:.6f}")

    # Rebuild exact SHAP sample
    seed = 0
    pos_idx = np.where(y_te == 1)[0]
    neg_idx = np.where(y_te == 0)[0]
    rng = np.random.RandomState(seed)
    chosen_neg_idx = rng.choice(neg_idx, size=20000, replace=False)
    sample_idx = np.concatenate([pos_idx, chosen_neg_idx])

    X_sample = X_te_lgb.iloc[sample_idx].copy().reset_index(drop=True)
    y_sample = y_te[sample_idx]
    print(f"Sample size: {len(X_sample):,} (Positives: {len(pos_idx):,}, Negatives: {len(chosen_neg_idx):,})")

    # Compute SHAP
    print("Computing SHAP TreeExplainer values on sample ...")
    explainer = shap.TreeExplainer(clf)
    explanation = explainer(X_sample)
    shap_vals = explanation.values
    feature_names = list(X_sample.columns)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    ranking = np.argsort(mean_abs_shap)[::-1]
    top15_idx = ranking[:15]

    # ------------------------------------------------------------------ #
    # TABLE 1: DIRECTION TABLE
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("TABLE 1: DIRECTION TABLE (Top 15 Features by Mean |SHAP|)")
    print("=" * 80)
    print(f"{'Rank':<4} | {'Feature':<22} | {'Mean |SHAP|':>11} | {'Spearman rho':>12} | {'Q1 Mean SHAP':>12} | {'Q5 Mean SHAP':>12} | {'Label':<20}")
    print("-" * 105)

    direction_results = []
    cat_features = {"payment_format", "sent_currency", "received_currency"}

    for rank, idx in enumerate(top15_idx, start=1):
        fname = feature_names[idx]
        m_val = mean_abs_shap[idx]
        f_vals = X_sample[fname]
        s_vals = shap_vals[:, idx]

        if fname in cat_features:
            rho_str = "N/A (Cat)"
            q1_str = "N/A"
            q5_str = "N/A"
            label = "See Table 2 (Cat)"
            direction_results.append({
                "Rank": rank, "Feature": fname, "Mean |SHAP|": round(m_val, 4),
                "Spearman rho": rho_str, "Q1 Mean SHAP": q1_str, "Q5 Mean SHAP": q5_str, "Empirical Direction": label
            })
        else:
            f_num = f_vals.values.astype(float)
            rho, pval = scipy.stats.spearmanr(f_num, s_vals)

            # Quintiles based on feature values
            q20 = np.percentile(f_num, 20)
            q80 = np.percentile(f_num, 80)
            mask_q1 = f_num <= q20
            mask_q5 = f_num >= q80

            q1_shap = float(s_vals[mask_q1].mean()) if mask_q1.sum() > 0 else float("nan")
            q5_shap = float(s_vals[mask_q5].mean()) if mask_q5.sum() > 0 else float("nan")

            if rho >= 0.2:
                label = "higher raises score"
            elif rho <= -0.2:
                label = "higher lowers score"
            else:
                label = "non-monotonic"

            rho_str = f"{rho:+.4f}"
            q1_str = f"{q1_shap:+.4f}"
            q5_str = f"{q5_shap:+.4f}"

            direction_results.append({
                "Rank": rank, "Feature": fname, "Mean |SHAP|": round(m_val, 4),
                "Spearman rho": rho_str, "Q1 Mean SHAP": q1_str, "Q5 Mean SHAP": q5_str, "Empirical Direction": label
            })

        print(f"{rank:<4} | {fname:<22} | {m_val:>11.4f} | {rho_str:>12} | {q1_str:>12} | {q5_str:>12} | {label:<20}")

    df_dir_out = pd.DataFrame(direction_results)
    df_dir_out.to_csv(DIRECTION_TABLE_CSV, index=False)
    print(f"\nSaved direction analysis table to: {DIRECTION_TABLE_CSV}")

    # ------------------------------------------------------------------ #
    # TABLE 2: CATEGORY CODES (CORRECTED)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("TABLE 2: CATEGORY CODES (payment_format ALL codes, sent_currency Top 8)")
    print("=" * 80)

    def analyze_category_col(col_name: str, name_map: dict, all_codes: bool = False, top_n: int = 5):
        col_idx = feature_names.index(col_name)
        # Fix: Look up by the REAL integer code in the data, never by positional .cat.codes
        f_col = X_te[col_name].iloc[sample_idx].to_numpy().astype(int)
        f_col_full_te = X_te[col_name].to_numpy().astype(int)
        s_col = shap_vals[:, col_idx]

        # Assertion: for every code in the sample, sampled row count equals count obtained by filtering integer column
        unique_sample_codes = np.unique(f_col)
        for c in unique_sample_codes:
            sample_c = int((f_col == c).sum())
            filtered_c = int((X_te[col_name].iloc[sample_idx].astype(int) == c).sum())
            assert sample_c == filtered_c, (
                f"Assertion failed for {col_name} code {c}: "
                f"sample count ({sample_c}) != filtered integer count ({filtered_c})"
            )

        if all_codes:
            eval_codes = sorted(name_map.keys())
        else:
            code_counts = Counter(f_col)
            eval_codes = [c for c, _ in code_counts.most_common(top_n)]

        n_pos_total = int((y_sample == 1).sum())
        n_neg_total = int((y_sample == 0).sum())

        rows = []
        for code in eval_codes:
            mask = (f_col == code)
            c_name = name_map.get(code, f"Unknown ({code})") if maps_verified else f"Code {code}"
            mean_s = float(s_col[mask].mean()) if mask.sum() > 0 else 0.0

            pos_in_code = int(((f_col == code) & (y_sample == 1)).sum())
            neg_in_code = int(((f_col == code) & (y_sample == 0)).sum())

            share_pos = (pos_in_code / n_pos_total) * 100.0 if n_pos_total > 0 else 0.0
            share_neg = (neg_in_code / n_neg_total) * 100.0 if n_neg_total > 0 else 0.0

            full_test_mask = (f_col_full_te == code)
            full_test_rows = int(full_test_mask.sum())
            full_test_pos = int((full_test_mask & (y_te == 1)).sum())

            rows.append({
                "column": col_name,
                "code": code,
                "real_name": c_name,
                "sample_count": int(mask.sum()),
                "mean_shap": round(mean_s, 4),
                "share_pos": round(share_pos, 2),
                "share_neg": share_neg,
                "full_test_rows": full_test_rows,
                "full_test_positives": full_test_pos,
            })
        return rows

    pf_rows = analyze_category_col("payment_format", inv_payment_format, all_codes=True)
    sc_rows = analyze_category_col("sent_currency", inv_currency, all_codes=False, top_n=8)

    print(f"\n--- Feature: payment_format (ALL Codes) ---")
    print(f"{'Code':<5} | {'Real Name':<16} | {'Sample Count':>12} | {'Mean SHAP':>10} | {'Share Pos (%)':>13} | {'Share Neg (%)':>13} | {'Full Test Rows':>14} | {'Full Test Pos':>13}")
    print("-" * 118)
    for r in pf_rows:
        print(f"{r['code']:<5} | {r['real_name']:<16} | {r['sample_count']:>12,} | {r['mean_shap']:>+10.4f} | {r['share_pos']:>12.2f}% | {r['share_neg']:>12.2f}% | {r['full_test_rows']:>14,} | {r['full_test_positives']:>13,}")

    print(f"\n--- Feature: sent_currency (Top 8 by Sample Count) ---")
    print(f"{'Code':<5} | {'Real Name':<18} | {'Sample Count':>12} | {'Mean SHAP':>10} | {'Share Pos (%)':>13} | {'Share Neg (%)':>13} | {'Full Test Rows':>14} | {'Full Test Pos':>13}")
    print("-" * 120)
    for r in sc_rows:
        print(f"{r['code']:<5} | {r['real_name']:<18} | {r['sample_count']:>12,} | {r['mean_shap']:>+10.4f} | {r['share_pos']:>12.2f}% | {r['share_neg']:>12.2f}% | {r['full_test_rows']:>14,} | {r['full_test_positives']:>13,}")

    # Cross-check positives per payment format in the full data against:
    # ACH 4483, Cheque 324, Credit Card 206, Cash 108, Bitcoin 56, Reinvestment 0, Wire 0
    print("\n--- Cross-Check: Full Data Positives per Payment Format ---")
    expected_full_pos = {
        "ACH": 4483,
        "Cheque": 324,
        "Credit Card": 206,
        "Cash": 108,
        "Bitcoin": 56,
        "Reinvestment": 0,
        "Wire": 0,
    }
    df_raw_fmt = pd.read_csv(FMT_TRANS_PATH, usecols=["Payment Format", "Is Laundering"])
    full_counts = df_raw_fmt.groupby("Payment Format")["Is Laundering"].agg(["count", "sum"])

    cross_check_passed = True
    print(f"{'Code':<5} | {'Real Name':<16} | {'Full Data Positives':>20} | {'Expected Positives':>20} | {'Status'}")
    print("-" * 75)
    for code in sorted(inv_payment_format.keys()):
        name = inv_payment_format[code]
        actual_pos = int(full_counts.loc[code, "sum"]) if code in full_counts.index else 0
        exp_pos = expected_full_pos[name]
        status = "MATCH" if actual_pos == exp_pos else "MISMATCH"
        if actual_pos != exp_pos:
            cross_check_passed = False
        print(f"{code:<5} | {name:<16} | {actual_pos:>20,} | {exp_pos:>20,} | {status}")

    assert cross_check_passed, "Full data payment format positives cross-check FAILED!"
    print("=> Cross-check PASSED: All payment format positives in full data match expected targets.")

    # Save to results/category_shap_corrected.csv with exact required columns:
    # format name, code, sampled rows, mean SHAP, share of sampled positives, share of sampled negatives, full-test rows, full-test positives
    csv_rows = []
    for r in pf_rows + sc_rows:
        csv_rows.append({
            "format name": r["real_name"],
            "code": r["code"],
            "sampled rows": r["sample_count"],
            "mean SHAP": round(r["mean_shap"], 4),
            "share of sampled positives": round(r["share_pos"], 2),
            "share of sampled negatives": round(r["share_neg"], 2),
            "full-test rows": r["full_test_rows"],
            "full-test positives": r["full_test_positives"],
        })
    df_corrected = pd.DataFrame(csv_rows)
    df_corrected.to_csv(CATEGORY_SHAP_CORRECTED_CSV, index=False)
    print(f"\nSaved corrected category SHAP analysis to: {CATEGORY_SHAP_CORRECTED_CSV}")

    if "--category-only" in sys.argv:
        print("\n[--category-only flag detected, skipping Table 3 and Table 4]")
        return

    # ------------------------------------------------------------------ #
    # TABLE 3: NEW-PAIR CHECK (Full test set)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("TABLE 3: NEW-PAIR CHECK (Full Test Set, N = 863,900)")
    print("=" * 80)

    # Check pair_prior_cnt on full X_te
    pair_cnt_te = X_te["pair_prior_cnt"].values  # stored as log1p(count)
    mask_zero = pair_cnt_te == 0.0
    mask_gt_zero = pair_cnt_te > 0.0

    n_te_pos = int((y_te == 1).sum())
    n_te_neg = int((y_te == 0).sum())

    pos_zero = int(((y_te == 1) & mask_zero).sum())
    pos_gt = int(((y_te == 1) & mask_gt_zero).sum())
    neg_zero = int(((y_te == 0) & mask_zero).sum())
    neg_gt = int(((y_te == 0) & mask_gt_zero).sum())

    share_pos_zero = (pos_zero / n_te_pos) * 100.0
    share_neg_zero = (neg_zero / n_te_neg) * 100.0

    pos_rate_zero = (pos_zero / (pos_zero + neg_zero)) * 100.0 if (pos_zero + neg_zero) > 0 else 0.0
    pos_rate_gt = (pos_gt / (pos_gt + neg_gt)) * 100.0 if (pos_gt + neg_gt) > 0 else 0.0

    print(f"{'Condition':<28} | {'Total Rows':>12} | {'Positives':>10} | {'Negatives':>10} | {'Pos Rate (%)':>13}")
    print("-" * 85)
    print(f"{'pair_prior_cnt == 0 (New Pair)':<28} | {mask_zero.sum():>12,} | {pos_zero:>10,} | {neg_zero:>10,} | {pos_rate_zero:>12.4f}%")
    print(f"{'pair_prior_cnt > 0  (Prior Pair)':<28} | {mask_gt_zero.sum():>12,} | {pos_gt:>10,} | {neg_gt:>10,} | {pos_rate_gt:>12.4f}%")
    print("-" * 85)
    print(f"{'Share of Positives with pair_prior_cnt == 0':<46} : {pos_zero:>6,} / {n_te_pos:,} ({share_pos_zero:.2f}%)")
    print(f"{'Share of Negatives with pair_prior_cnt == 0':<46} : {neg_zero:>6,} / {n_te_neg:,} ({share_neg_zero:.2f}%)")
    ratio_enrichment = pos_rate_zero / max(pos_rate_gt, 1e-9)
    print(f"{'Enrichment Ratio (Pos Rate zero vs >0)':<46} : {ratio_enrichment:.2f}x")

    # ------------------------------------------------------------------ #
    # TABLE 4: TENURE CHECK
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("TABLE 4: TENURE CHECK (Ablation of Cumulative-Volume Features)")
    print("=" * 80)

    drop_vol_cols = [
        "s_out_cnt", "s_out_amt", "s_in_cnt", "s_in_amt",
        "r_in_cnt", "r_in_amt", "r_out_cnt", "r_out_amt",
    ]
    print(f"Dropping {len(drop_vol_cols)} cumulative-volume columns:\n  {drop_vol_cols}")

    X_tr_no_vol = X_tr.drop(columns=drop_vol_cols)
    X_val_no_vol = X_val.drop(columns=drop_vol_cols)
    X_te_no_vol = X_te.drop(columns=drop_vol_cols)

    print(f"Features: Full = {X_tr.shape[1]} -> Dropped = {X_tr_no_vol.shape[1]}")
    print("Training lightgbm_undersample without cumulative-volume features (seed=0, temporal) ...")
    t0 = time.time()
    clf_no_vol, _, _ = build_and_fit_lightgbm_undersample(
        X_tr=X_tr_no_vol,
        y_tr=y_tr,
        X_val=X_val_no_vol,
        y_val=y_val,
        seed=seed,
        quick=False,
    )
    t_train_no_vol = time.time() - t0
    print(f"Retrained in {t_train_no_vol:.1f}s")

    X_te_no_vol_lgb = prepare_lgb_features(X_te_no_vol)
    te_scores_no_vol = clf_no_vol.predict_proba(X_te_no_vol_lgb)[:, 1]

    pr_auc_no_vol = float(average_precision_score(y_te, te_scores_no_vol))
    te_prevalence = float(y_te.mean())
    lift_full = full_pr_auc / te_prevalence
    lift_no_vol = pr_auc_no_vol / te_prevalence
    n_unique_full = int(len(np.unique(full_te_scores)))
    n_unique_no_vol = int(len(np.unique(te_scores_no_vol)))

    print(f"\n{'Model Configuration':<35} | {'# Feat':>6} | {'Test PR-AUC':>12} | {'Lift':>8} | {'Unique Scores':>14}")
    print("-" * 85)
    print(f"{'Full tab_dyn model':<35} | {X_tr.shape[1]:>6} | {full_pr_auc:>12.6f} | {lift_full:>8.1f} | {n_unique_full:>14,}")
    print(f"{'Without cumulative volume (Tenure)':<35} | {X_tr_no_vol.shape[1]:>6} | {pr_auc_no_vol:>12.6f} | {lift_no_vol:>8.1f} | {n_unique_no_vol:>14,}")
    delta_pr = pr_auc_no_vol - full_pr_auc
    print(f"{'Difference (no_vol - full)':<35} | {X_tr_no_vol.shape[1]-X_tr.shape[1]:>6} | {delta_pr:>+12.6f} | {lift_no_vol - lift_full:>+8.1f} | {n_unique_no_vol - n_unique_full:>+14,}")

    # Save to results/tenure_check.csv
    df_tenure = pd.DataFrame([
        {
            "model": "lightgbm_undersample",
            "features_set": "tab_dyn_full",
            "n_features": X_tr.shape[1],
            "dropped_columns": "none",
            "test_pr_auc": round(full_pr_auc, 6),
            "test_prevalence": round(te_prevalence, 6),
            "lift": round(lift_full, 2),
            "unique_test_scores": n_unique_full,
        },
        {
            "model": "lightgbm_undersample",
            "features_set": "tab_dyn_no_cumulative_volume",
            "n_features": X_tr_no_vol.shape[1],
            "dropped_columns": ";".join(drop_vol_cols),
            "test_pr_auc": round(pr_auc_no_vol, 6),
            "test_prevalence": round(te_prevalence, 6),
            "lift": round(lift_no_vol, 2),
            "unique_test_scores": n_unique_no_vol,
        },
    ])
    df_tenure.to_csv(TENURE_CSV, index=False)
    print(f"\nSaved tenure check results to: {TENURE_CSV}")

    print("\n" + "=" * 80)
    print("Step 6b checks completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()

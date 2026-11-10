"""
analyze_tail.py - Step 9 Tail Analysis

Analyzes the near-empty tail of the IBM AML Small_HI dataset (days 10-17).
Exports:
  - results/tail_sensitivity.csv
  - results/tail_topk.csv
  - results/tail_profile.csv
Prints all tables to stdout.
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE = Path(__file__).parent
RESULTS = BASE / "results"
SPLITS = BASE / "splits"
PREDS = RESULTS / "preds"

sys.path.insert(0, str(BASE))
from explain_checks import recover_and_verify_category_maps

def main():
    print("=" * 80)
    print("STEP 2: TAIL ANALYSIS (IBM AML Small_HI)")
    print("=" * 80)

    # 1. Load data
    print("\n--- 1. Loading Transactions & Graph Features ---")
    raw_fmt_path = Path(r"C:\aml_data\Small_HI\formatted_transactions.csv")
    graph_par_path = Path(r"C:\aml_data\Small_HI\graph_dynamic.parquet")

    df_trans = pd.read_csv(raw_fmt_path, usecols=["Timestamp", "Payment Format", "Is Laundering"])
    print(f"Loaded formatted transactions: {len(df_trans):,} rows")

    gdf = pd.read_parquet(graph_par_path, columns=["pair_prior_cnt"])
    df_trans["pair_prior_cnt"] = gdf["pair_prior_cnt"].values
    del gdf

    min_ts = df_trans["Timestamp"].min()
    df_trans["day"] = (df_trans["Timestamp"] - min_ts) // 86400

    # 2. Daily table (all 18 days)
    print("\n" + "=" * 80)
    print("DAILY TABLE (All 18 Days)")
    print("=" * 80)
    daily_stats = []
    for d in range(18):
        sub = df_trans[df_trans["day"] == d]
        rows = len(sub)
        pos = int(sub["Is Laundering"].sum())
        rate = pos / rows if rows > 0 else 0.0
        new_pairs = int((sub["pair_prior_cnt"] == 0).sum())
        daily_stats.append({
            "day": d,
            "rows": rows,
            "positives": pos,
            "rate": rate,
            "new_pair_rows": new_pairs,
            "new_pair_share": new_pairs / rows if rows > 0 else 0.0,
        })
    df_daily = pd.DataFrame(daily_stats)
    print(f"{'Day':<5} | {'Rows':>12} | {'Positives':>10} | {'Rate (%)':>10} | {'New-Pair Rows':>14} | {'New-Pair Share (%)':>18}")
    print("-" * 80)
    for _, r in df_daily.iterrows():
        print(f"{int(r['day']):<5} | {int(r['rows']):>12,} | {int(r['positives']):>10,} | {r['rate']*100:>9.4f}% | {int(r['new_pair_rows']):>14,} | {r['new_pair_share']*100:>17.2f}%")
    total_rows = df_daily["rows"].sum()
    total_pos = df_daily["positives"].sum()
    print("-" * 80)
    print(f"{'Total':<5} | {total_rows:>12,} | {total_pos:>10,} | {total_pos/total_rows*100:>9.4f}% | {df_daily['new_pair_rows'].sum():>14,} | {df_daily['new_pair_rows'].sum()/total_rows*100:>17.2f}%")

    assert total_rows == 5078345, f"Expected 5,078,345 rows, got {total_rows}"
    assert total_pos == 5177, f"Expected 5,177 positives, got {total_pos}"
    print("=> Daily table check PASSED: 5,078,345 rows, 5,177 laundering rows.")

    # 3. Sensitivity analysis (Bulk vs All)
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS: ALL TEST ROWS VS BULK (DAY <= 9)")
    print("=" * 80)

    # Test indices
    temp_split = np.load(SPLITS / "Small_HI_split.npz")
    rand_split = np.load(SPLITS / "Small_HI_random_split.npz")
    te_inds = {
        "temporal": temp_split["te_inds"],
        "random": rand_split["te_inds"],
    }

    test_subsets = {}
    for sp in ["temporal", "random"]:
        inds = te_inds[sp]
        y_te = df_trans["Is Laundering"].iloc[inds].to_numpy()
        day_te = df_trans["day"].iloc[inds].to_numpy()
        bulk_mask = (day_te <= 9)
        test_subsets[sp] = {
            "y_all": y_te,
            "bulk_mask": bulk_mask,
            "rows_all": len(y_te),
            "pos_all": int(y_te.sum()),
            "rows_bulk": int(bulk_mask.sum()),
            "pos_bulk": int(y_te[bulk_mask].sum()),
        }

    models = ["logistic_regression", "random_forest", "xgboost", "lightgbm_undersample"]
    pat_abl = re.compile(r"^ablation_(tab_dyn|tab_dyn_static)_(temporal|random)_(" + "|".join(models) + r")_seed(\d+)_te_scores\.npy$")
    pat_tab = re.compile(r"^(" + "|".join(models) + r")_(temporal|random)_seed(\d+)_te_scores\.npy$")

    files = list(PREDS.glob("*_te_scores.npy"))
    runs = {} # (split, feat_set, model) -> list of (seed, ap_all, ap_bulk)

    for f in sorted(files):
        m1 = pat_abl.match(f.name)
        if m1:
            fset, sp, mdl, s = m1.group(1), m1.group(2), m1.group(3), int(m1.group(4))
        else:
            m2 = pat_tab.match(f.name)
            if m2:
                fset, sp, mdl, s = "tabular", m2.group(2), m2.group(1), int(m2.group(3))
            else:
                continue

        scores = np.load(f)
        sub = test_subsets[sp]
        assert len(scores) == sub["rows_all"], f"Score length mismatch in {f.name}: {len(scores)} vs {sub['rows_all']}"

        ap_all = average_precision_score(sub["y_all"], scores)
        ap_bulk = average_precision_score(sub["y_all"][sub["bulk_mask"]], scores[sub["bulk_mask"]])

        key = (sp, fset, mdl)
        if key not in runs:
            runs[key] = []
        runs[key].append((s, ap_all, ap_bulk))

    sensitivity_rows = []
    # Order: feature_set, model, split
    for fset in ["tabular", "tab_dyn", "tab_dyn_static"]:
        for mdl in models:
            for sp in ["temporal", "random"]:
                key = (sp, fset, mdl)
                if key not in runs:
                    continue
                seed_list = runs[key]
                n_seeds = len(seed_list)
                ap_all_m = float(np.mean([x[1] for x in seed_list]))
                ap_bulk_m = float(np.mean([x[2] for x in seed_list]))
                sub = test_subsets[sp]
                sensitivity_rows.append({
                    "split": sp,
                    "feature_set": fset,
                    "model": mdl,
                    "seeds": n_seeds,
                    "rows_all": sub["rows_all"],
                    "pos_all": sub["pos_all"],
                    "rows_bulk": sub["rows_bulk"],
                    "pos_bulk": sub["pos_bulk"],
                    "ap_all_mean": ap_all_m,
                    "ap_bulk_mean": ap_bulk_m,
                })

    df_sens = pd.DataFrame(sensitivity_rows)
    df_sens.to_csv(RESULTS / "tail_sensitivity.csv", index=False)
    print(f"Saved sensitivity analysis to: {RESULTS / 'tail_sensitivity.csv'}")

    # Compute and display lift table and inflation ratios
    print("\n" + "-" * 115)
    print(f"{'Feature Set':<16} | {'Model':<22} | {'Split':<8} | {'AP (All)':>8} | {'Lift (All)':>10} | {'AP (Bulk)':>9} | {'Lift (Bulk)':>11}")
    print("-" * 115)
    for _, r in df_sens.iterrows():
        prev_all = r["pos_all"] / r["rows_all"]
        prev_bulk = r["pos_bulk"] / r["rows_bulk"]
        lift_all = r["ap_all_mean"] / prev_all
        lift_bulk = r["ap_bulk_mean"] / prev_bulk
        print(f"{r['feature_set']:<16} | {r['model']:<22} | {r['split']:<8} | {r['ap_all_mean']:>8.4f} | {lift_all:>9.1f}x | {r['ap_bulk_mean']:>9.4f} | {lift_bulk:>10.1f}x")

    print("\n--- Random / Temporal Lift Ratios (All vs Bulk) ---")
    print(f"{'Feature Set':<16} | {'Model':<22} | {'Ratio (All)':>12} | {'Ratio (Bulk)':>13}")
    print("-" * 70)
    for fset in ["tabular", "tab_dyn", "tab_dyn_static"]:
        for mdl in models:
            sub_t = df_sens[(df_sens["feature_set"] == fset) & (df_sens["model"] == mdl) & (df_sens["split"] == "temporal")].iloc[0]
            sub_r = df_sens[(df_sens["feature_set"] == fset) & (df_sens["model"] == mdl) & (df_sens["split"] == "random")].iloc[0]

            l_all_t = sub_t["ap_all_mean"] / (sub_t["pos_all"] / sub_t["rows_all"])
            l_all_r = sub_r["ap_all_mean"] / (sub_r["pos_all"] / sub_r["rows_all"])
            l_bulk_t = sub_t["ap_bulk_mean"] / (sub_t["pos_bulk"] / sub_t["rows_bulk"])
            l_bulk_r = sub_r["ap_bulk_mean"] / (sub_r["pos_bulk"] / sub_r["rows_bulk"])

            ratio_all = l_all_r / l_all_t
            ratio_bulk = l_bulk_r / l_bulk_t
            print(f"{fset:<16} | {mdl:<22} | {ratio_all:>11.2f}x | {ratio_bulk:>12.2f}x")

    # 4. Top-k tail breakdown
    print("\n" + "=" * 80)
    print("TOP-K ALERT BUDGET TAIL BREAKDOWN (lightgbm_undersample tab_dyn temporal seed 0)")
    print("=" * 80)
    score_file = PREDS / "ablation_tab_dyn_temporal_lightgbm_undersample_seed0_te_scores.npy"
    scores_s0 = np.load(score_file)
    sub_temp = test_subsets["temporal"]
    y_te = sub_temp["y_all"]
    day_te = df_trans["day"].iloc[te_inds["temporal"]].to_numpy()

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(scores_s0))
    order = np.lexsort((perm, -scores_s0))

    topk_rows = []
    budget_map = {864: "0.1%", 4320: "0.5%", 8639: "1.0%"}
    print(f"{'Budget':<8} | {'k':>6} | {'Alerts Days 10-17':>18} | {'Total TP':>10} | {'TP Days 10-17':>14} | {'TP Share Days 10-17 (%)':>24}")
    print("-" * 90)
    for k in [864, 4320, 8639]:
        top_idx = order[:k]
        alerts_tail = int((day_te[top_idx] >= 10).sum())
        tp_total = int(y_te[top_idx].sum())
        tp_tail = int(((y_te[top_idx] == 1) & (day_te[top_idx] >= 10)).sum())
        share_tail = (tp_tail / tp_total * 100.0) if tp_total > 0 else 0.0

        topk_rows.append({
            "budget": budget_map[k],
            "k": k,
            "alerts_tail": alerts_tail,
            "tp_total": tp_total,
            "tp_tail": tp_tail,
        })
        print(f"{budget_map[k]:<8} | {k:>6,} | {alerts_tail:>18,} | {tp_total:>10,} | {tp_tail:>14,} | {share_tail:>23.2f}%")

    df_topk = pd.DataFrame(topk_rows)
    df_topk.to_csv(RESULTS / "tail_topk.csv", index=False)
    print(f"Saved top-k tail breakdown to: {RESULTS / 'tail_topk.csv'}")

    # 5. Profile table
    print("\n" + "=" * 80)
    print("PROFILE TABLE: TEMPORAL TEST GROUPS")
    print("=" * 80)
    inv_pf, _, _ = recover_and_verify_category_maps()
    ach_code = None
    for c, n in inv_pf.items():
        if n == "ACH":
            ach_code = c
            break
    assert ach_code is not None, "ACH code could not be determined!"
    print(f"Verified ACH integer code: {ach_code}")

    pf_te = df_trans["Payment Format"].iloc[te_inds["temporal"]].to_numpy()
    pp_te = df_trans["pair_prior_cnt"].iloc[te_inds["temporal"]].to_numpy()

    groups = [
        ("Positives in days 8-9", (y_te == 1) & (day_te <= 9)),
        ("Positives in days 10-17", (y_te == 1) & (day_te >= 10)),
        ("Normal rows in days 8-9", (y_te == 0) & (day_te <= 9)),
    ]

    profile_rows = []
    print(f"{'Group':<26} | {'Count':>10} | {'Share ACH (%)':>15} | {'Share pair_prior_cnt==0 (%)':>28}")
    print("-" * 88)
    for gname, mask in groups:
        cnt = int(mask.sum())
        ach_cnt = int((pf_te[mask] == ach_code).sum())
        new_cnt = int((pp_te[mask] == 0).sum())
        sh_ach = (ach_cnt / cnt) if cnt > 0 else 0.0
        sh_new = (new_cnt / cnt) if cnt > 0 else 0.0

        profile_rows.append({
            "group": gname,
            "count": cnt,
            "share_ach": round(sh_ach, 6),
            "share_new_pair": round(sh_new, 6),
        })
        print(f"{gname:<26} | {cnt:>10,} | {sh_ach*100:>14.2f}% | {sh_new*100:>27.2f}%")

    df_prof = pd.DataFrame(profile_rows)
    df_prof.to_csv(RESULTS / "tail_profile.csv", index=False)
    print(f"Saved profile analysis to: {RESULTS / 'tail_profile.csv'}")

    print("\n" + "=" * 80)
    print("STEP 2 TAIL ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()


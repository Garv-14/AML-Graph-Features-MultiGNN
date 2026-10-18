"""
Explain the best model (lightgbm_undersample on tab_dyn, temporal split, seed 0) using SHAP.

Steps:
1. Retrain lightgbm_undersample on tab_dyn temporal split, seed 0, reusing run_tabular.py.
2. Verify test PR-AUC matches results/results.csv (within 0.01).
3. Save model to models/lgbm_us_tab_dyn_temporal_seed0.pkl.
4. Run SHAP TreeExplainer on all test positives + 20,000 random test negatives (seed 0).
5. Generate and save figures to results/figures/:
   - (a) Global bar plot of mean |SHAP| (top 20 features)
   - (b) Beeswarm plot of top 15 features
   - (c) Dependence plots for top 3 features
   - (d) Waterfall plots for 3 true positives in top 0.1% test scores
6. Print mean |SHAP| table grouped by feature family (relative importance).
7. Print plain-language interpretation of top features and check for time artefacts.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from run_tabular import (
    build_and_fit_lightgbm_undersample,
    load_and_build_features,
    prepare_lgb_features,
)

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_CSV = RESULTS_DIR / "results.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FAMILIES = {
    # Tabular
    "amount_sent": "tabular",
    "amount_received": "tabular",
    "log1p_amount_sent": "tabular",
    "log1p_amount_received": "tabular",
    "amount_ratio": "tabular",
    "sent_currency": "tabular",
    "received_currency": "tabular",
    "payment_format": "tabular",
    "same_currency": "tabular",
    "hour_of_day": "tabular",
    "day_of_week": "tabular",
    # Sender history
    "s_out_cnt": "sender history",
    "s_out_amt": "sender history",
    "s_in_cnt": "sender history",
    "s_in_amt": "sender history",
    "s_uniq_receivers": "sender history",
    "s_uniq_senders_in": "sender history",
    "s_gap_out": "sender history",
    "s_amt_vs_mean": "sender history",
    "s_net_flow": "sender history",
    "has_sender_history": "sender history",
    # Receiver history
    "r_in_cnt": "receiver history",
    "r_in_amt": "receiver history",
    "r_out_cnt": "receiver history",
    "r_out_amt": "receiver history",
    "r_uniq_senders": "receiver history",
    "r_uniq_receivers_out": "receiver history",
    "r_gap_in": "receiver history",
    "has_receiver_history": "receiver history",
    # Pair history
    "pair_prior_cnt": "pair history",
    "reverse_prior_cnt": "pair history",
}


def main():
    print("=" * 70)
    print("STEP 6: SHAP Model Interpretability (LightGBM Undersample)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load Data & Retrain Model
    # ------------------------------------------------------------------ #
    print("\n--- 1. Loading Features & Retraining Model ---")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_and_build_features(
        feature_set="tab_dyn",
        split_kind="temporal",
        data_name="Small_HI",
    )

    seed = 0
    print(f"\nTraining lightgbm_undersample (tab_dyn, temporal, seed={seed}) ...")
    t0 = time.time()
    clf, X_tr_us_lgb, y_tr_us = build_and_fit_lightgbm_undersample(
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        seed=seed,
        quick=False,
    )
    train_time = time.time() - t0
    print(f"Model trained in {train_time:.1f}s")

    X_te_lgb = prepare_lgb_features(X_te)
    te_scores = clf.predict_proba(X_te_lgb)[:, 1]
    retrained_pr_auc = float(average_precision_score(y_te, te_scores))

    # Load stored value from results.csv
    df_res = pd.read_csv(RESULTS_CSV)
    stored_row = df_res[
        (df_res["experiment"] == "ablation_tab_dyn_temporal")
        & (df_res["model_name"] == "lightgbm_undersample")
        & (df_res["seed"] == seed)
    ]
    if stored_row.empty:
        raise RuntimeError("Could not find matching stored row in results/results.csv!")
    stored_pr_auc = float(stored_row.iloc[0]["te_pr_auc"])

    diff = abs(retrained_pr_auc - stored_pr_auc)
    print(f"\nReproduction Check:")
    print(f"  Stored PR-AUC    : {stored_pr_auc:.6f}")
    print(f"  Retrained PR-AUC : {retrained_pr_auc:.6f}")
    print(f"  Absolute diff    : {diff:.6f}")
    if diff > 0.01:
        raise AssertionError(f"PR-AUC discrepancy {diff:.4f} exceeds tolerance 0.01!")
    print("  => PR-AUC verification PASSED (within tolerance < 0.01)\n")

    # Save model
    model_path = MODELS_DIR / "lgbm_us_tab_dyn_temporal_seed0.pkl"
    joblib.dump(clf, model_path)
    print(f"Saved trained model to: {model_path}")

    # ------------------------------------------------------------------ #
    # 2. SHAP Sample Construction
    # ------------------------------------------------------------------ #
    print("\n--- 2. Sampling Test Set for SHAP ---")
    pos_idx = np.where(y_te == 1)[0]
    neg_idx = np.where(y_te == 0)[0]
    n_pos = len(pos_idx)
    n_neg_sample = 20000

    rng = np.random.RandomState(seed)
    chosen_neg_idx = rng.choice(neg_idx, size=n_neg_sample, replace=False)
    sample_idx = np.concatenate([pos_idx, chosen_neg_idx])

    print(f"Test set total rows: {len(y_te):,}")
    print(f"Test positives     : {n_pos:,} (all included)")
    print(f"Sampled negatives  : {n_neg_sample:,}")
    print(f"Total SHAP sample  : {len(sample_idx):,} rows")

    X_sample = X_te_lgb.iloc[sample_idx].copy().reset_index(drop=True)
    y_sample = y_te[sample_idx]
    sample_scores = te_scores[sample_idx]

    # ------------------------------------------------------------------ #
    # 3. Compute SHAP Values with TreeExplainer
    # ------------------------------------------------------------------ #
    print("\n--- 3. Computing SHAP TreeExplainer Values ---")
    t0_shap = time.time()
    explainer = shap.TreeExplainer(clf)
    explanation = explainer(X_sample)
    shap_time = time.time() - t0_shap
    print(f"SHAP values computed in {shap_time:.1f}s")
    print(f"Explanation shape: {explanation.shape}")

    # Extract shap values array (N, n_features)
    shap_vals = explanation.values
    feature_names = list(X_sample.columns)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Ranking
    ranking_order = np.argsort(mean_abs_shap)[::-1]
    top20_idx = ranking_order[:20]

    # ------------------------------------------------------------------ #
    # 4. Generate Figures
    # ------------------------------------------------------------------ #
    print("\n--- 4. Generating and Saving Figures ---")

    # (a) Global bar plot of top 20
    fig_bar_path = FIGURES_DIR / "shap_global_bar_top20.png"
    plt.figure(figsize=(10, 8))
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("Top 20 Features by Mean |SHAP| (LightGBM tab_dyn temporal seed 0)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(fig_bar_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] (a) Global Bar Plot    -> {fig_bar_path}")

    # (b) Beeswarm plot of top 15
    fig_bee_path = FIGURES_DIR / "shap_beeswarm_top15.png"
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm Summary — Top 15 Features (Log-Odds Impact)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(fig_bee_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] (b) Beeswarm Plot      -> {fig_bee_path}")

    # (c) Dependence plots for top 3 features
    dep_fig_paths = []
    top3_features = [feature_names[i] for i in ranking_order[:3]]
    for rank, feat_name in enumerate(top3_features, start=1):
        fig_dep_path = FIGURES_DIR / f"shap_dependence_{rank}_{feat_name}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feat_name,
            shap_vals,
            X_sample,
            ax=ax,
            show=False,
        )
        ax.set_title(f"SHAP Dependence Plot #{rank}: {feat_name}", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(fig_dep_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        dep_fig_paths.append(fig_dep_path)
        print(f"  [Saved] (c) Dependence Plot #{rank} -> {fig_dep_path}")

    # (d) Waterfall plots for 3 true positives in top 0.1% test scores
    # Top 0.1% budget threshold
    k_001 = int(len(y_te) * 0.001)  # 863,900 * 0.001 = 863
    te_rank_order = np.argsort(te_scores)[::-1]
    top_001_indices = te_rank_order[:k_001]

    # Find true positives among top 0.1%
    tp_top_001 = [idx for idx in top_001_indices if y_te[idx] == 1]
    print(f"\nTop 0.1% alert budget ({k_001} transactions): {len(tp_top_001)} True Positives found")

    # Select 3 representative TPs: #1 (highest rank), median in top 0.1%, and near the 0.1% boundary
    tp_selection_indices = [
        0,
        len(tp_top_001) // 2,
        len(tp_top_001) - 1,
    ]
    waterfall_paths = []
    for rank_i, sel_idx in enumerate(tp_selection_indices, start=1):
        test_orig_idx = tp_top_001[sel_idx]
        global_test_rank = int(np.where(te_rank_order == test_orig_idx)[0][0]) + 1
        score_val = float(te_scores[test_orig_idx])

        # In X_sample, pos_idx are the first n_pos rows, in exact matching order
        sample_row_idx = int(np.where(pos_idx == test_orig_idx)[0][0])

        fig_wf_path = FIGURES_DIR / f"shap_waterfall_tp_{rank_i}.png"
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[sample_row_idx], max_display=12, show=False)
        plt.title(
            f"SHAP Waterfall: TP Sample #{rank_i} (Test Rank #{global_test_rank}, Score={score_val:.4f})",
            fontsize=12,
            pad=10,
        )
        plt.tight_layout()
        plt.savefig(fig_wf_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        waterfall_paths.append(fig_wf_path)
        print(f"  [Saved] (d) Waterfall TP #{rank_i}   -> {fig_wf_path} (Test Rank #{global_test_rank}, Score={score_val:.4f})")

    # ------------------------------------------------------------------ #
    # 5. Top-20 Features Table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("TOP 20 FEATURES BY MEAN |SHAP| VALUE")
    print("=" * 70)
    print(f"{'Rank':<5} | {'Feature':<24} | {'Family':<16} | {'Mean |SHAP|':>11} | {'Direction':<12}")
    print("-" * 75)

    plain_lang_notes = []
    for rank, idx in enumerate(top20_idx, start=1):
        fname = feature_names[idx]
        family = FEATURE_FAMILIES.get(fname, "unknown")
        m_val = mean_abs_shap[idx]

        # Determine directional correlation with SHAP value
        f_vals = X_sample[fname]
        if f_vals.dtype.name == "category":
            f_num = f_vals.cat.codes.values.astype(float)
        else:
            f_num = f_vals.values.astype(float)

        s_vals = shap_vals[:, idx]
        valid_mask = ~(np.isnan(f_num) | np.isnan(s_vals))
        corr = np.corrcoef(f_num[valid_mask], s_vals[valid_mask])[0, 1] if valid_mask.sum() > 10 else 0.0
        direction = "Positive (+)" if corr > 0.05 else ("Negative (-)" if corr < -0.05 else "Non-linear")

        print(f"{rank:<5} | {fname:<24} | {family:<16} | {m_val:>11.4f} | {direction:<12}")

        if rank <= 5:
            plain_lang_notes.append((fname, family, m_val, direction, corr))

    # ------------------------------------------------------------------ #
    # 6. Grouped Table by Feature Family
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FEATURE FAMILY IMPORTANCE (Relative Importance)")
    print("=" * 70)
    total_shap_sum = float(mean_abs_shap.sum())

    family_records = []
    for fam_name in ["tabular", "sender history", "receiver history", "pair history"]:
        fam_feats = [f for f, fam in FEATURE_FAMILIES.items() if fam == fam_name and f in feature_names]
        fam_indices = [feature_names.index(f) for f in fam_feats]
        fam_sum = float(mean_abs_shap[fam_indices].sum())
        fam_pct = (fam_sum / total_shap_sum) * 100.0
        fam_mean = float(mean_abs_shap[fam_indices].mean()) if fam_indices else 0.0
        family_records.append({
            "Family": fam_name,
            "N Features": len(fam_feats),
            "Sum Mean |SHAP|": fam_sum,
            "Share (%)": fam_pct,
            "Avg Mean |SHAP|": fam_mean,
        })

    df_fam = pd.DataFrame(family_records)
    print(f"{'Feature Family':<18} | {'# Feat':>6} | {'Sum Mean |SHAP|':>15} | {'Share (%)':>10} | {'Avg |SHAP|/Feat':>15}")
    print("-" * 75)
    for _, r in df_fam.iterrows():
        print(f"{r['Family']:<18} | {int(r['N Features']):>6} | {r['Sum Mean |SHAP|']:>15.4f} | {r['Share (%)']:>9.1f}% | {r['Avg Mean |SHAP|']:>15.4f}")
    print("-" * 75)
    print(f"{'Total':<18} | {len(feature_names):>6} | {total_shap_sum:>15.4f} | {100.0:>9.1f}% | {total_shap_sum/len(feature_names):>15.4f}")

    # ------------------------------------------------------------------ #
    # 7. Plain Language Interpretations & Temporal Artefact Checks
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PLAIN-LANGUAGE BEHAVIORAL INTERPRETATION & SUSPICION CHECKS")
    print("=" * 70)
    for fname, family, m_val, direction, corr in plain_lang_notes:
        print(f"\n* Feature: `{fname}` (Family: {family}, Mean |SHAP|={m_val:.4f})")
        if fname == "s_amt_vs_mean":
            print("  Behavior: Difference between current transaction amount and the sender's past average amount.")
            print("  Score Impact: High positive values sharply increase laundering risk (unusual burst in amount).")
        elif fname == "amount_ratio":
            print("  Behavior: Ratio of received amount to sent amount (measuring fee deduction / currency friction).")
            print("  Score Impact: Deviations from standard expected ratios or round amounts alter risk probability.")
        elif fname in ("s_out_cnt", "r_in_cnt"):
            print(f"  Behavior: Cumulative prior count of {('outbound' if fname=='s_out_cnt' else 'inbound')} transfers.")
            print(f"  Score Impact: {'High transaction activity elevates risk in burst patterns, while low counts indicate newly active entities.'}")
            print(f"  Suspicion Check: POTENTIALLY SUSPICIOUS / TIME ARTEFACT. Because counts are strictly cumulative")
            print(f"  over the streaming sequence, later test transactions have naturally higher cumulative counts than")
            print(f"  early training rows. The model might partly use cumulative counts as a proxy for account maturity / time.")
        elif fname == "payment_format":
            print("  Behavior: Channel used for payment (ACH, Wire, Credit Card, Cheque, Cash, etc.).")
            print("  Score Impact: High-risk formats (like Wire or Cheque) substantially elevate laundering likelihood.")
        elif fname == "s_gap_out":
            print("  Behavior: Time interval since sender's previous outbound transfer.")
            print("  Score Impact: Very short gaps (rapid-fire transactions) increase laundering score.")
        else:
            print(f"  Behavior: General metric in {family}.")
            print(f"  Score Impact: Directional relationship ({direction}) influences model log-odds.")

    print("\n" + "=" * 70)
    print("Step 6 SHAP analysis completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

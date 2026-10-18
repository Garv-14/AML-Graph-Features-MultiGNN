"""Tabular model training, threshold selection, and evaluation for IBM AML.

Trains six model families on the temporal or random split, selects the best
F1 threshold on validation, evaluates on test, and logs results to
results/results.csv via evaluate.py.

Models
------
logistic_regression   : Sklearn Pipeline with ColumnTransformer (OneHotEncoder + StandardScaler), class_weight='balanced'
random_forest         : RandomForestClassifier (n_estimators=100/30, min_samples_leaf=20, max_samples=0.2, balanced_subsample)
lightgbm              : LGBMClassifier with category dtypes, scale_pos_weight, and PR-AUC early stopping (30 rounds)
xgboost               : XGBClassifier with tree_method='hist', scale_pos_weight, and PR-AUC early stopping (30 rounds)
isolation_forest      : IsolationForest (contamination='auto', raw -score_samples, no min-max or clipping)
lightgbm_undersample  : LGBMClassifier on 50:1 neg:pos ratio train set, no scale_pos_weight, PR-AUC early stopping

Usage
-----
  python run_tabular.py --quick --seeds 0
  python run_tabular.py --split_kind temporal --seeds 0 1 2
  python run_tabular.py --split_kind random --seeds 0 1 2
  python run_tabular.py --models lightgbm --seeds 0
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_access import load_split
from evaluate import alert_budget_metrics, append_result, best_f1_threshold, evaluate_scores
from features_tabular import FEATURE_COLS, build_tabular_features

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"
PREDS_DIR = RESULTS_DIR / "preds"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_CSV = RESULTS_DIR / "results.csv"

RESULTS_DIR.mkdir(exist_ok=True)
PREDS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Registry of supported model names
# ---------------------------------------------------------------------------
ALL_MODELS = [
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "xgboost",
    "isolation_forest",
    "lightgbm_undersample",
]

ABLATION_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm_undersample",
]

LABEL_COL = "Is Laundering"
CAT_COLS = ["sent_currency", "received_currency", "payment_format"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


# ---------------------------------------------------------------------------
# Undersampling helper
# ---------------------------------------------------------------------------

def _undersample_50_to_1(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    rng: np.random.RandomState,
    neg_to_pos_ratio: float = 50.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Random majority undersampling of train negatives to a 50:1 neg:pos ratio.

    Keeps all positives and randomly samples negatives to reach exactly
    neg_to_pos_ratio * n_positives.

    Args:
        X_train: Training feature DataFrame.
        y_train: Binary label vector.
        rng: NumPy RandomState for reproducibility.
        neg_to_pos_ratio: Target negative to positive ratio (default 50.0).

    Returns:
        Tuple (X_resampled, y_resampled) with shuffled rows.
    """
    pos_mask = y_train == 1
    n_pos = int(pos_mask.sum())
    n_neg_target = int(n_pos * neg_to_pos_ratio)

    neg_idx = np.where(~pos_mask)[0]
    if n_neg_target < len(neg_idx):
        chosen_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
    else:
        chosen_neg = neg_idx

    pos_idx = np.where(pos_mask)[0]
    combined = np.concatenate([pos_idx, chosen_neg])
    rng.shuffle(combined)
    return X_train.iloc[combined].reset_index(drop=True), y_train[combined]


# ---------------------------------------------------------------------------
# Core: train, select threshold, evaluate one model
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    seed: int,
    split_kind: str,
    quick: bool,
    save_model: bool,
    experiment_name: str | None = None,
) -> dict:
    """Train a model, pick the best F1 threshold on val, evaluate on test.

    Args:
        model_name: One of ALL_MODELS.
        X_tr / y_tr: Training features (DataFrame) and labels.
        X_val / y_val: Validation features and labels (threshold selection & early stopping).
        X_te / y_te: Test features and labels (final evaluation).
        seed: Random seed.
        split_kind: 'temporal' or 'random' (used in output file names).
        quick: If True, use reduced estimator counts for fast smoke tests.
        save_model: If True, persist fitted model to MODELS_DIR (seed 0 only).
        experiment_name: Optional experiment identifier for results.csv.

    Returns:
        Dict with experiment identifiers and all metric values for results.csv.
    """
    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    rng = np.random.RandomState(seed)

    print(f"  [{model_name}] seed={seed}  train_rows={len(y_tr):,}  pos={n_pos:,}  features={X_tr.shape[1]}")
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Fit and score
    # ------------------------------------------------------------------ #
    if model_name == "logistic_regression":
        num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                ("num", StandardScaler(), num_cols),
            ]
        )
        clf = Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=300,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ])
        clf.fit(X_tr, y_tr)
        val_scores = clf.predict_proba(X_val)[:, 1]
        te_scores = clf.predict_proba(X_te)[:, 1]

    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=30 if quick else 100,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            max_samples=0.2,
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
        val_scores = clf.predict_proba(X_val)[:, 1]
        te_scores = clf.predict_proba(X_te)[:, 1]

    elif model_name == "lightgbm":
        X_tr_lgb = X_tr.copy()
        X_val_lgb = X_val.copy()
        X_te_lgb = X_te.copy()
        for col in CAT_COLS:
            X_tr_lgb[col] = X_tr_lgb[col].astype("category")
            X_val_lgb[col] = X_val_lgb[col].astype("category")
            X_te_lgb[col] = X_te_lgb[col].astype("category")

        scale = float(np.sqrt(float(n_neg) / max(float(n_pos), 1.0)))
        clf = lgb.LGBMClassifier(
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=100 if quick else 400,
            scale_pos_weight=scale,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            metric="None",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        clf.fit(
            X_tr_lgb,
            y_tr,
            eval_set=[(X_val_lgb, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        val_scores = clf.predict_proba(X_val_lgb)[:, 1]
        te_scores = clf.predict_proba(X_te_lgb)[:, 1]
        if save_model:
            joblib.dump(clf, MODELS_DIR / f"lgbm_{split_kind}_seed{seed}.pkl")

    elif model_name == "xgboost":
        scale = float(n_neg) / max(float(n_pos), 1.0)
        clf = xgb.XGBClassifier(
            tree_method="hist",
            scale_pos_weight=scale,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            n_estimators=100 if quick else 400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        val_scores = clf.predict_proba(X_val)[:, 1]
        te_scores = clf.predict_proba(X_te)[:, 1]
        if save_model:
            joblib.dump(clf, MODELS_DIR / f"xgb_{split_kind}_seed{seed}.pkl")

    elif model_name == "isolation_forest":
        clf = IsolationForest(
            n_estimators=50 if quick else 200,
            contamination="auto",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_tr)
        val_scores = -clf.score_samples(X_val)
        te_scores = -clf.score_samples(X_te)

    elif model_name == "lightgbm_undersample":
        X_tr_us, y_tr_us = _undersample_50_to_1(X_tr, y_tr, rng=rng, neg_to_pos_ratio=50.0)
        X_tr_us_lgb = X_tr_us.copy()
        X_val_lgb = X_val.copy()
        X_te_lgb = X_te.copy()
        for col in CAT_COLS:
            X_tr_us_lgb[col] = X_tr_us_lgb[col].astype("category")
            X_val_lgb[col] = X_val_lgb[col].astype("category")
            X_te_lgb[col] = X_te_lgb[col].astype("category")

        clf = lgb.LGBMClassifier(
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=100 if quick else 400,
            scale_pos_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            metric="None",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        clf.fit(
            X_tr_us_lgb,
            y_tr_us,
            eval_set=[(X_val_lgb, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        val_scores = clf.predict_proba(X_val_lgb)[:, 1]
        te_scores = clf.predict_proba(X_te_lgb)[:, 1]

    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    train_time = time.time() - t0
    print(f"  [{model_name}] trained in {train_time:.1f}s")

    # ------------------------------------------------------------------ #
    # Unique scores check and warning
    # ------------------------------------------------------------------ #
    n_unique_te = int(len(np.unique(te_scores)))
    print(f"  [{model_name}] seed={seed} unique test scores: {n_unique_te:,}")
    if n_unique_te < 1000:
        warnings.warn(
            f"Model {model_name} (seed {seed}) has fewer than 1000 unique test scores ({n_unique_te})!"
        )

    # ------------------------------------------------------------------ #
    # Threshold selection on VALIDATION only (no test leakage)
    # ------------------------------------------------------------------ #
    threshold = best_f1_threshold(y_val, val_scores)

    # ------------------------------------------------------------------ #
    # Evaluate on both splits at the chosen threshold
    # ------------------------------------------------------------------ #
    val_metrics = evaluate_scores(
        y_val, val_scores, threshold, model_name=model_name, split_name="val"
    )
    te_metrics = evaluate_scores(
        y_te, te_scores, threshold, model_name=model_name, split_name="test"
    )
    budget_metrics = alert_budget_metrics(y_te, te_scores)

    # ------------------------------------------------------------------ #
    # Persist prediction arrays for downstream analysis / ensembling
    # ------------------------------------------------------------------ #
    exp_name = experiment_name if experiment_name else f"tabular_{split_kind}"
    tag = f"{exp_name}_{model_name}_seed{seed}"
    np.save(PREDS_DIR / f"{tag}_val_scores.npy", val_scores)
    np.save(PREDS_DIR / f"{tag}_te_scores.npy", te_scores)

    # ------------------------------------------------------------------ #
    # Build flat result record for results.csv
    # ------------------------------------------------------------------ #
    record: dict = {
        "experiment": exp_name,
        "model_name": model_name,
        "split_kind": split_kind,
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "threshold": threshold,
        "te_unique_scores": n_unique_te,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    for k, v in val_metrics.items():
        if k not in ("model_name", "split_name"):
            record[f"val_{k}"] = v
    for k, v in te_metrics.items():
        if k not in ("model_name", "split_name"):
            record[f"te_{k}"] = v
    for budget_key, bdict in budget_metrics.items():
        for bk, bv in bdict.items():
            record[f"te_{budget_key}_{bk}"] = bv

    # Human-readable one-line summary
    print(
        f"  [{model_name}] seed={seed} | "
        f"val PR-AUC={val_metrics['pr_auc']:.4f} | "
        f"te  PR-AUC={te_metrics['pr_auc']:.4f}  "
        f"ROC-AUC={te_metrics['roc_auc']:.4f}  "
        f"F1={te_metrics['f1']:.4f}  "
        f"P={te_metrics['precision']:.4f}  R={te_metrics['recall']:.4f}"
    )

    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train tabular AML baselines and append results to results/results.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split_kind",
        choices=["temporal", "random"],
        default="temporal",
        help="Split type: 'temporal' (chronological) or 'random'.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds. Runs all models for each seed independently.",
    )
    parser.add_argument(
        "--feature_set",
        choices=["tabular", "tab_dyn", "tab_dyn_static"],
        default="tabular",
        help="Feature set to use: 'tabular' (baseline), 'tab_dyn', or 'tab_dyn_static'.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=None,
        help="Subset of model names to run (defaults to ALL_MODELS for tabular, ABLATION_MODELS for ablation).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: reduced estimator counts for fast smoke testing.",
    )
    parser.add_argument(
        "--data",
        default="Small_HI",
        help="Dataset name matching a sub-directory under the AML data root.",
    )
    args = parser.parse_args()

    # Determine models to run
    if args.models is None:
        models_to_run = ALL_MODELS if args.feature_set == "tabular" else ABLATION_MODELS
    else:
        models_to_run = args.models

    experiment_name = (
        f"tabular_{args.split_kind}"
        if args.feature_set == "tabular"
        else f"ablation_{args.feature_set}_{args.split_kind}"
    )

    print(
        f"\n=== Experiment: {experiment_name} | feature_set={args.feature_set}  "
        f"split={args.split_kind}  seeds={args.seeds}  quick={args.quick} ==="
    )
    print(f"Models to run: {models_to_run}")

    # ------------------------------------------------------------------ #
    # Load split (identical across seeds)
    # ------------------------------------------------------------------ #
    print("Loading split ...")
    df_train, df_val, df_test = load_split(data_name=args.data, kind=args.split_kind)

    y_tr = df_train[LABEL_COL].values.astype(int)
    y_val_arr = df_val[LABEL_COL].values.astype(int)
    y_te = df_test[LABEL_COL].values.astype(int)

    pct_tr = 100.0 * y_tr.mean()
    pct_val = 100.0 * y_val_arr.mean()
    pct_te = 100.0 * y_te.mean()
    print(f"  train={len(y_tr):,}  pos={y_tr.sum():,}  ({pct_tr:.3f}%)")
    print(f"  val  ={len(y_val_arr):,}  pos={y_val_arr.sum():,}  ({pct_val:.3f}%)")
    print(f"  test ={len(y_te):,}  pos={y_te.sum():,}  ({pct_te:.3f}%)")

    # ------------------------------------------------------------------ #
    # Build features once (no randomness; same for all seeds)
    # ------------------------------------------------------------------ #
    print("Building base tabular features ...")
    X_tr = build_tabular_features(df_train)
    X_val_feat = build_tabular_features(df_val)
    X_te_feat = build_tabular_features(df_test)

    # ------------------------------------------------------------------ #
    # Load and concatenate extra graph features if ablation feature set
    # ------------------------------------------------------------------ #
    if args.feature_set != "tabular":
        from data_access import get_data_dir

        data_dir = Path(get_data_dir()) / args.data
        splits_dir = Path(__file__).parent / "splits"
        split_file = f"{args.data}_split.npz" if args.split_kind == "temporal" else f"{args.data}_random_split.npz"
        sp_indices = np.load(splits_dir / split_file)
        tr_inds = sp_indices["tr_inds"]
        val_inds = sp_indices["val_inds"]
        te_inds = sp_indices["te_inds"]

        # Load dynamic features
        dyn_path = data_dir / "graph_dynamic.parquet"
        print(f"Loading dynamic graph features from {dyn_path} ...")
        df_dyn = pd.read_parquet(dyn_path)
        dyn_tr = df_dyn.iloc[tr_inds].reset_index(drop=True)
        dyn_val = df_dyn.iloc[val_inds].reset_index(drop=True)
        dyn_te = df_dyn.iloc[te_inds].reset_index(drop=True)

        if args.feature_set == "tab_dyn":
            X_tr = pd.concat([X_tr, dyn_tr], axis=1)
            X_val_feat = pd.concat([X_val_feat, dyn_val], axis=1)
            X_te_feat = pd.concat([X_te_feat, dyn_te], axis=1)
        elif args.feature_set == "tab_dyn_static":
            stat_path = data_dir / f"graph_static_{args.split_kind}.parquet"
            print(f"Loading static graph features from {stat_path} ...")
            df_stat = pd.read_parquet(stat_path)
            stat_tr = df_stat.iloc[tr_inds].reset_index(drop=True)
            stat_val = df_stat.iloc[val_inds].reset_index(drop=True)
            stat_te = df_stat.iloc[te_inds].reset_index(drop=True)
            X_tr = pd.concat([X_tr, dyn_tr, stat_tr], axis=1)
            X_val_feat = pd.concat([X_val_feat, dyn_val, stat_val], axis=1)
            X_te_feat = pd.concat([X_te_feat, dyn_te, stat_te], axis=1)

    print(
        f"  Feature DataFrame shapes:  "
        f"train={X_tr.shape}  val={X_val_feat.shape}  test={X_te_feat.shape}"
    )

    # ------------------------------------------------------------------ #
    # Outer loop: seeds x models
    # ------------------------------------------------------------------ #
    all_records: list[dict] = []
    for seed in args.seeds:
        print(f"\n--- seed={seed} ---")
        for mname in models_to_run:
            save_model = seed == 0  # Persist trained models only for seed 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = run_one_model(
                    model_name=mname,
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_val=X_val_feat,
                    y_val=y_val_arr,
                    X_te=X_te_feat,
                    y_te=y_te,
                    seed=seed,
                    split_kind=args.split_kind,
                    quick=args.quick,
                    save_model=save_model,
                    experiment_name=experiment_name,
                )
            append_result(record, path=str(RESULTS_CSV))
            all_records.append(record)

    # ------------------------------------------------------------------ #
    # Final summary table
    # ------------------------------------------------------------------ #
    print(f"\n=== Summary (split={args.split_kind}) ===")
    print(f"{'Model':<26} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 68)
    for r in all_records:
        print(
            f"{r['model_name']:<26} "
            f"{r.get('te_pr_auc', float('nan')):>8.4f} "
            f"{r.get('te_roc_auc', float('nan')):>8.4f} "
            f"{r.get('te_f1', float('nan')):>7.4f} "
            f"{r.get('te_precision', float('nan')):>7.4f} "
            f"{r.get('te_recall', float('nan')):>7.4f}"
        )
    print(f"\nResults appended to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()

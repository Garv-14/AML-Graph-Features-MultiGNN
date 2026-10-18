# Step 2 Changes Log: IBM/Multi-GNN Adaptation for AML

This document details every modification and new file introduced in Step 2 to remove startup blockers, make the repository pip-runnable on Windows and Colab, and export shared dataset splits.

---

## Summary of Changes

| File | Status | Description of Change | Rationale |
|---|---|---|---|
| `requirements.txt` | **Created** | Pinned common data science, ML, and GNN utility packages (`numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `networkx`, `tabulate`, `wandb`, etc.) without hardcoding PyTorch/PyG. | Enables standard `pip install -r requirements.txt` across Windows and Colab without Conda lock-ins. |
| `SETUP.md` | **Created** | Comprehensive instructions for Windows CPU/CUDA, Google Colab GPU runtime, verification of PyG imports without `torch-scatter`/`sparse`, and dataset setup. | Clear, reproducible installation and environment instructions for project team members. |
| `format_kaggle_files.py` | **Modified** | Replaced `datatable` with Python `csv` streaming reader and `pandas` for timestamp sorting and CSV output. Fixed path joining with `os.path.join`. | Eliminates `datatable` build failures on modern Python/Windows, preserves exact column order and encoding dictionaries. |
| `data_config.json` | **Modified** | Set default `aml_data` to `"C:/aml_data"` and relative model folders (`./saved_models/`). | Replaces `/path/to/...` placeholder values that triggered immediate file not found errors. |
| `data_loading.py` | **Modified** | Added `os.environ.get("AML_DATA_DIR", ...)` fallback and `os.path.join` path resolution. | Allows dynamic configuration of dataset root directory without modifying code files. |
| `train_util.py` | **Modified** | Changed `file_path = './model_settings.json'` to use `os.path.dirname(os.path.abspath(__file__))`. | Prevents `FileNotFoundError` when invoking scripts from outside the repository root directory. |
| `util.py` | **Modified** | Added `--use_wandb` flag (default `False`). Changed `--unique_name` argument type from `action='store_true'` to `default="", type=str`. | Prevents mandatory online wandb authentication, fixes string interpolation bug in model checkpoint naming. |
| `training.py` | **Modified** | Set default WandB mode to `"disabled"` unless `--use_wandb` is explicitly provided and not in `--testing` mode. Renamed project from `"your_proj_name"` to `"aml-minor-project"`. | Allows seamless offline execution and provides descriptive project tagging. |
| `inference.py` | **Modified** | Added header warning comment: `KNOWN BROKEN: calls evaluate_homo(precrec=True) and uses args.avg_tps; replaced by evaluate.py`. Updated wandb project and default disabled mode. | Flags upstream repository bugs in inference routine to prevent confusion. |
| `export_splits.py` | **Created** | Utility script to load `Small_HI`, extract `tr_inds`, `val_inds`, `te_inds`, calculate day cutoffs `(i, j)`, export `.npz` and `.json` summaries to `splits/`, and print summary table. | Provides shared, reproducible dataset splits for training non-GNN baseline models (XGBoost, LightGBM, Random Forest). |
| `.baseline_original/` | **Created** | Direct local backup directory containing untouched copies of all original repository files. | Serves as a local diff snapshot and backup because git could not be initialized without system git binaries. |

---

## Model Logic & Reproducibility Guarantee
- Model architectures (`GINe`, `GATe`, `PNA`, `RGCN`) in `models.py` were **not modified**.
- Loss functions, sampling parameters, neighbor counts, and optimizer logic in `training.py` and `train_util.py` were **not modified**.
- Preprocessing mathematics (Z-score normalization, port assignments, and daily temporal partitioning) were **not modified**.

## Key Environmental & Technical Finding
- **PyG Neighbor Sampling Dependency**: While PyG static imports succeed without C++ extensions, PyG's `LinkNeighborLoader` relies on `NeighborSampler`, which at runtime calls `pyg-lib` or `torch-sparse`. On Windows with Python 3.13, pre-compiled wheels for `pyg-lib` do not yet exist on PyPI and compiling `torch-sparse` requires MSVC. On Google Colab or Linux with Python 3.10/3.11, pre-built binary wheels install seamlessly. Full split extraction (`export_splits.py`) and data formatting (`format_kaggle_files.py`) work cleanly and without dependency on C++ extensions.

---

# Step 3 Changes Log: Shared Evaluation Module & Random Split Export

This step introduced a centralized, model-agnostic evaluation framework and exported a random split partition to benchmark temporal drift against random cross-validation. No existing GNN code (`main.py`, `training.py`, `train_util.py`, `data_loading.py`, `models.py`) was modified.

## Summary of Changes (Step 3)

| File | Status | Description of Change | Rationale |
|---|---|---|---|
| `evaluate.py` | **Created** | Shared evaluation module containing `evaluate_scores` (PR-AUC, ROC-AUC, F1, precision, recall, confusion matrix), `best_f1_threshold` (optimal operating threshold on validation PR curve), `alert_budget_metrics` (precision@k and recall@k under compliance alert capacity), and `append_result` (persists experiment runs to CSV). | Unifies performance evaluation across tabular, graph-feature, and GNN pipelines under consistent AML metrics. |
| `data_access.py` | **Created** | Central data loading module with `load_formatted` (reads `formatted_transactions.csv` via configurable data directory) and `load_split` (loads temporal or random splits as DataFrames using saved `.npz` indices). | Ensures all models train and evaluate on strictly identical row sequences. |
| `make_random_split.py` | **Created** | Script generating a random split of identical partition sizes (3,248,921 train / 965,524 val / 863,900 test) with fixed seed 42, saved to `splits/Small_HI_random_split.npz` and summary JSON. | Enables the "Temporal vs. Random Split" experiment to isolate the impact of temporal drift. |
| `tests/test_evaluate.py` | **Created** | Unit test suite verifying perfect scorer edge cases, random scorer convergence to prevalence, threshold optimization, single-class handling, and alert budget metric correctness. | Ensures metric implementations are mathematically verified and robust against edge cases. |
| `sanity_check.py` | **Created** | Integration script verifying zero index overlap (disjoint sets) across both temporal and random splits, validating DataFrame loading, and displaying comparative laundering rates. | Guarantees split integrity and prevents data leakage across partitions. |


---

## Step 4 – Tabular Baselines

| File | Status | Description of Change | Rationale |
|---|---|---|---|
| `features_tabular.py` | **Created** | Per-row feature engineering module exporting `FEATURE_COLS` and `build_tabular_features(df)`. Produces 11 features: `amount_sent`, `amount_received`, `log1p_amount_sent`, `log1p_amount_received`, `amount_ratio`, `sent_currency`, `received_currency`, `payment_format`, `same_currency`, `hour_of_day`, `day_of_week`. No cross-row aggregation; safe against temporal leakage. Explicitly excludes `EdgeID`, `from_id`, `to_id`, raw `Timestamp`, and `Is Laundering`. | Provides a clean, reusable feature set shared identically by all six tabular models and compatible with both temporal and random splits. |
| `run_tabular.py` | **Created** | End-to-end training and evaluation script for six model families: `logistic_regression` (class_weight=balanced), `random_forest` (n=300, balanced), `lightgbm` (scale_pos_weight), `xgboost` (scale_pos_weight), `isolation_forest` (unsupervised, contamination from train prevalence), `lightgbm_undersample` (random majority undersampling to 10% positives). Threshold selected exclusively on validation PR curve via `best_f1_threshold`; then applied unchanged to test. Saves predictions as `.npy` to `results/preds/` and fitted models (seed 0 only) as `.pkl` to `models/`. Results appended to `results/results.csv` via `append_result`. CLI accepts `--split_kind {temporal,random}`, `--seeds`, `--models`, `--quick`, `--data`. | Central training harness for the tabular phase of the project, designed to produce both the main temporal result and the contrast random-split experiment. |
| `summarize_results.py` | **Created** | Reads `results/results.csv`, filters tabular experiments, computes mean ± std across seeds for PR-AUC, ROC-AUC, F1, Precision, Recall, and alert-budget metrics (P@0.1%, P@0.5%, P@1%). Prints and saves `results/summary_tabular.md`. | Produces publication-ready comparison tables for both temporal and random splits. |
| `.gitignore` | **Updated** | Added `results/preds/`, `models/`, `*.pkl`, `*.npy` exclusions. `results/results.csv` remains tracked. | Prevents large binary prediction and model artefacts from entering version control while keeping the summary CSV in git history. |

### Runs performed

| Command | Purpose |
|---|---|
| `python run_tabular.py --quick --seeds 0` | Smoke test (reduced estimators, all 6 models) |
| `python run_tabular.py --split_kind temporal --seeds 0 1 2` | Main result: temporal split, 3 seeds |
| `python run_tabular.py --split_kind random --seeds 0 1 2` | Contrast experiment: random split, 3 seeds |
| `python summarize_results.py` | Generates `results/summary_tabular.md` |

---

## Step 4b – Model Alignment, Ties Fix, and Re-Run

This step resolves model specification discrepancies from Step 4, fixes row-order bias in the alert-budget evaluation metric, and produces refreshed end-to-end benchmarks for both temporal and random splits across seeds 0, 1, and 2.

### Summary of Changes (Step 4b)

| File | Status | Description of Change | Rationale |
|---|---|---|---|
| `evaluate.py` | **Modified** | In `alert_budget_metrics`, equal predicted scores are now broken with a seeded random permutation (seed 0) via stable argsort prior to taking the top $k$ items. | Eliminates row-order dependence in alert budget evaluations (preventing artificial 100% precision when positive rows cluster at the start of a split). |
| `tests/test_evaluate.py` | **Modified** | Added `test_alert_budget_metrics_ties_and_shuffling` verifying: (1) on a large synthetic dataset with identical scores, precision@k is close to prevalence; and (2) when scores are distinct, shuffling input rows yields identical metric results. | Verifies mathematically sound tie-breaking and order invariance. |
| `run_tabular.py` | **Modified** | Re-implemented models to strictly match original specs: <br>• `logistic_regression`: Scikit-Learn `Pipeline` with `ColumnTransformer` (One-hot encoding `sent_currency`, `received_currency`, `payment_format` with `handle_unknown="ignore"`, and `StandardScaler` on all numeric features fitted strictly on train), `class_weight="balanced"`, `max_iter=300`.<br>• `random_forest`: `n_estimators=100` (30 when `--quick`), `min_samples_leaf=20`, `class_weight="balanced_subsample"`, `max_samples=0.2`, `n_jobs=-1`.<br>• `lightgbm`: `sent_currency`, `received_currency`, `payment_format` encoded as pandas `category` dtype; `num_leaves=63`, `learning_rate=0.05`, `n_estimators=400` with early stopping on validation split (`eval_metric="average_precision"`, `stopping_rounds=30`), `scale_pos_weight = negatives/positives`.<br>• `xgboost`: `tree_method="hist"`, `scale_pos_weight = negatives/positives`, `eval_metric="aucpr"`, `early_stopping_rounds=30` on validation split, `n_estimators=400`.<br>• `lightgbm_undersample`: Random majority undersampling to a 50:1 negative-to-positive ratio (preserving all positives), `scale_pos_weight=1.0`, same early stopping as LightGBM.<br>• `isolation_forest`: `contamination="auto"` (no label-derived values), fitted on train features only; uses raw `-score_samples` without min-max scaling or clipping.<br>• Added unique score tracking and warning when test scores have fewer than 1000 unique values. | Aligns all tabular models with baseline specifications, fixes convergence failure in logistic regression, stabilizes early stopping, and tracks score discretization. |
| `summarize_results.py` | **Modified** | Generates side-by-side comparison tables between temporal and random splits; computes PR-AUC lift (`PR-AUC / test prevalence`), ROC-AUC, F1, Precision, and Recall (mean ± std); prints test prevalence on its own line; outputs full Alert Budget tables (P@k and R@k for 0.1%, 0.5%, 1.0%); reports mean training times per model in seconds; writes to `results/summary_tabular.md`. | Delivers complete, publication-ready comparative metrics isolating temporal distribution drift. |

### Backups & Artifacts
- Previous results moved: `results/results.csv` -> `results/results_v1_backup.csv`, `results/summary_tabular.md` -> `results/summary_v1_backup.md`.
- Stale prediction arrays in `results/preds/` and checkpoints in `models/` were cleared before running fresh experiments.

### Step 4b Runs Performed

| Command | Purpose |
|---|---|
| `python tests/test_evaluate.py` | Executed shared evaluation unit test suite including tie-breaking verification. |
| `python run_tabular.py --quick --seeds 0` | Smoke test verifying all 6 model pipelines end-to-end. |
| `python run_tabular.py --split_kind temporal --seeds 0 1 2` | Full temporal split run across 3 seeds. |
| `python run_tabular.py --split_kind random --seeds 0 1 2` | Full random split run across 3 seeds. |
| `python summarize_results.py` | Generated final comparative markdown report in `results/summary_tabular.md`. |

---

## Step 4c – LightGBM Metric Calibration and Re-Run

This step diagnoses and addresses early stopping and score collapse in plain LightGBM and aligns the metric monitoring for both LightGBM variants.

### Summary of Changes (Step 4c)

| File | Status | Description of Change | Rationale |
|---|---|---|---|
| `run_tabular.py` | **Modified** | For both `lightgbm` and `lightgbm_undersample`, added `metric="None"` to `LGBMClassifier` parameters so that default `binary_logloss` is omitted and early stopping monitors exclusively `eval_metric="average_precision"`. Switched plain `lightgbm` from `scale_pos_weight = negatives/positives` (~1283) to `scale_pos_weight = sqrt(negatives/positives)` (~35.8). | Prevents dual-metric premature early stopping conflict where unweighted validation logloss degraded rapidly under extreme class weighting, eliminates ranking collapse (temporal ROC-AUC was 0.088 on seed 2 in Step 4b), and stabilizes PR-AUC from 0.012 to 0.069 on temporal split. |

### Backups & Runs Performed
- Backed up `results/results.csv` to `results/results_v2_backup.csv`.
- Deleted existing rows for `lightgbm` and `lightgbm_undersample` from `results/results.csv`.
- Executed `run_tabular.py --split_kind temporal --seeds 0 1 2 --models lightgbm lightgbm_undersample`.
- Executed `run_tabular.py --split_kind random --seeds 0 1 2 --models lightgbm lightgbm_undersample`.
- Executed `summarize_results.py` to refresh `results/summary_tabular.md`.




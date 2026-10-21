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

---

## Step 5 — Leakage-Safe Graph Features & Ablation

### Goal
Build vectorised, leakage-free dynamic and static graph features on top of the tabular
baseline and run a full ablation (tabular vs tab+dyn vs tab+dyn+static) on both the
temporal and random splits.

### Design Rules (enforced)
- Rows of `formatted_transactions.csv` are already chronologically ordered (non-decreasing
  Timestamp verified at runtime). Row position is the universal transaction id.
- **Dynamic features** for row `i` use ONLY rows with position `< i` (no future leakage).
  Same-second ties resolved by row position.
- All dynamic features are computed from the **full stream** (no split-specific logic).
- **Static features** (PageRank, in-/out-degree) are computed **only from the training rows**
  of each split to prevent test-set leakage.
- `from_id`, `to_id`, `EdgeID`, and raw `Timestamp` are never used as model inputs.
- Heavy-tailed features stored as `log1p`; missing history stored as `0` with explicit
  `has_sender_history` / `has_receiver_history` flags.

### New / Modified Files

| File | Status | Description |
|------|--------|-------------|
| `graph_features.py` | **Created** | Vectorised dynamic feature builder (`build_dynamic_features`) using `np.searchsorted` + prefix sums; sparse weighted PageRank (`compute_pagerank`) via `scipy.sparse`; static degree builder (`build_static_features`). Saves outputs to `C:\aml_data\Small_HI\graph_dynamic.parquet`, `graph_static_temporal.parquet`, `graph_static_random.parquet`. |
| `tests/test_graph_features.py` | **Created** | Unit tests: Naive Check, Future Invariance (zero leakage), No-Self/Label, PageRank convergence. All 4 tests pass. |
| `run_tabular.py` | **Modified** | Added `--feature_set {tabular,tab_dyn,tab_dyn_static}` CLI option; `ABLATION_MODELS` registry; parquet loading/concatenation; experiment naming `ablation_<feature_set>_<split_kind>`. |
| `summarize_ablation.py` | **Created** | Reads `results/results.csv`, aggregates over 3 seeds, writes `results/summary_ablation.md`. |
| `results/summary_ablation.md` | **Created** | Full ablation summary with PR-AUC +/- std, ROC-AUC, F1, P@0.1%, P@0.5%, and delta PR-AUC vs tabular baseline. |
| `results/results_v3_backup.csv` | **Created** | Backup of `results.csv` before Step 5 ablation rows were appended. |

### Feature Engineering Details

**Dynamic features (20 per row, Group A+B):**
- Sender: log1p count, sum, mean, std of past amounts; time-since-last; has_sender_history
- Receiver: same 6 features for receiver side
- Cross-pair: log1p pair-count and pair-sum for (sender, receiver); has_pair_history
- Temporal recency: rolling 1-hour and 24-hour log1p counts for sender and receiver (4 features)

**Static features (6 per row, Group C — train-only):**
- Sender PageRank, Receiver PageRank
- Sender in-/out-degree, Receiver in-/out-degree

**History coverage (temporal split):**
- Train: Sender 13.22% / Receiver 11.44% no-history
- Val:   Sender  0.09% / Receiver  0.02% no-history
- Test:  Sender  0.10% / Receiver  0.02% no-history

Feature build runtime: ~24 s dynamic, ~5 s static per split. Memory: ~775 MB RAM peak.

### Ablation Results (test-set PR-AUC, mean +/- std, 3 seeds)

#### Temporal split

| Feature set       | LR     | RF     | XGB    | LGBM   |
|-------------------|--------|--------|--------|--------|
| tabular (11)      | 0.0287 | 0.1702 | 0.1274 | 0.1723 |
| tab+dyn (31)      | 0.3379 | 0.6644 | 0.5726 | **0.7080** |
| tab+dyn+stat (37) | 0.1542 | 0.6525 | 0.5817 | 0.6858 |

Δ vs tabular (LightGBM): tab+dyn = +0.54, tab+dyn+stat = +0.51

#### Random split

| Feature set       | LR     | RF     | XGB    | LGBM   |
|-------------------|--------|--------|--------|--------|
| tabular (11)      | 0.0147 | 0.0764 | 0.0769 | 0.0865 |
| tab+dyn (31)      | 0.1973 | 0.5362 | 0.5490 | 0.6057 |
| tab+dyn+stat (37) | 0.2061 | 0.5487 | 0.5517 | **0.6178** |

Δ vs tabular (LightGBM): tab+dyn = +0.52, tab+dyn+stat = +0.53

**Best — temporal:** tab+dyn + LightGBM -> PR-AUC 0.7080 +/- 0.0129
**Best — random:** tab+dyn+static + LightGBM -> PR-AUC 0.6178 +/- 0.0013

### Runs Performed
- Backed up `results/results.csv` to `results/results_v3_backup.csv`.
- Full ablation runs (seeds 0, 1, 2):
  - `run_tabular.py --feature_set tab_dyn --split_kind temporal`
  - `run_tabular.py --feature_set tab_dyn --split_kind random`
  - `run_tabular.py --feature_set tab_dyn_static --split_kind temporal`
  - `run_tabular.py --feature_set tab_dyn_static --split_kind random`

---

## Step 6 — Model Interpretability with SHAP

### Goal
Explain the top-performing model (`lightgbm_undersample` on `tab_dyn` temporal split, seed 0) using SHAP (`TreeExplainer`) to uncover global feature importance, feature family contributions, and local predictions on true positive laundering transactions.

### Changes & New Files

| File | Status | Description |
|------|--------|-------------|
| `explain_shap.py` | **Created** | Script to retrain `lightgbm_undersample` on `tab_dyn` temporal split (seed 0), verify exact PR-AUC against `results.csv`, save model checkpoint, compute SHAP values over all test positives + 20k sampled negatives, generate interpretability figures, and output family/feature tables. |
| `run_tabular.py` | **Modified** | Refactored `prepare_lgb_features`, `build_and_fit_lightgbm_undersample`, and `load_and_build_features` as clean, reusable functions imported directly by `explain_shap.py`. |
| `models/lgbm_us_tab_dyn_temporal_seed0.pkl` | **Created** | Serialized trained LightGBM model for seed 0. |
| `results/figures/shap_global_bar_top20.png` | **Created** | Global bar chart of mean \|SHAP\| across the top 20 features. |
| `results/figures/shap_beeswarm_top15.png` | **Created** | Summary beeswarm plot of log-odds impact for top 15 features. |
| `results/figures/shap_dependence_1_pair_prior_cnt.png` | **Created** | Dependence plot for #1 feature (`pair_prior_cnt`). |
| `results/figures/shap_dependence_2_payment_format.png` | **Created** | Dependence plot for #2 feature (`payment_format`). |
| `results/figures/shap_dependence_3_s_gap_out.png` | **Created** | Dependence plot for #3 feature (`s_gap_out`). |
| `results/figures/shap_waterfall_tp_1.png` | **Created** | Local waterfall plot for top test TP (Rank #1 overall, score 0.9998). |
| `results/figures/shap_waterfall_tp_2.png` | **Created** | Local waterfall plot for median test TP in top 0.1% (Rank #402, score 0.9952). |
| `results/figures/shap_waterfall_tp_3.png` | **Created** | Local waterfall plot for boundary test TP in top 0.1% (Rank #863, score 0.9768). |

### Key Findings & Metrics
- **Exact PR-AUC Reproduction**: Stored = `0.694059`, Retrained = `0.694059` (diff = `0.000000`, well within `< 0.01`).
- **Feature Family Relative Importance**:
  - `sender history`: **31.8%** share of total \|SHAP\| (10 features, avg 0.1979/feat).
  - `pair history`: **28.3%** share (2 features, avg 0.8808/feat — highest per-feature density).
  - `tabular`: **26.8%** share (11 features, avg 0.1517/feat).
  - `receiver history`: **13.1%** share (8 features, avg 0.1018/feat).
- **Behavioral Insights & Suspicion Checks**:
  - `pair_prior_cnt` is the #1 feature: high past interaction frequency sharply *decreases* laundering probability (negative correlation), identifying recurring legitimate business ties vs suspicious first-time pairs.
  - `payment_format` is #2: high-friction/high-value channels (e.g. Wire, Cheque) strongly elevate risk.
  - `s_gap_out` is #3: rapid transaction bursts (very small gaps) push scores up.
  - *Time Artefact Warning*: Cumulative counters (`s_out_cnt`, `r_in_cnt`) grow monotonically over time; while informative of burst velocity, they partially function as proxies for account age/tenure over chronological streams.

---

## Step 6b — Empirical Verification of SHAP Claims & Quantification

### Goal
Replace qualitative hypotheses and unverified guesses from the initial SHAP write-up with exact empirical measurements, verified category code mappings, and an ablation of cumulative-volume counters.

### New Files

| File | Status | Description |
|------|--------|-------------|
| `explain_checks.py` | **Created** | Script implementing direction analysis (Spearman rho + quintile SHAPs), category code decoding and frequency verification against raw data, full-test new-pair frequency check, and tenure ablation. |
| `results/tenure_check.csv` | **Created** | Benchmark comparison of full `tab_dyn` model (31 features) vs tenure-ablated model (23 features, dropping 8 cumulative-volume counters). |

### Key Findings & Empirical Tables

1. **Direction Analysis (Top 15 Features by Mean \|SHAP\|)**:
   - `pair_prior_cnt` (Rank 1, \|SHAP\|=1.6890): Spearman $\rho = -0.8947$, Q1 = $+1.6698$, Q5 = $-2.2976$ $\rightarrow$ **higher lowers score**.
   - `s_gap_out` (Rank 3, \|SHAP\|=0.5995): Spearman $\rho = +0.2372$, Q1 = $+0.3911$, Q5 = $+1.6730$ $\rightarrow$ **higher raises score**.
   - `s_uniq_receivers` (Rank 4, \|SHAP\|=0.5213): Spearman $\rho = -0.1701$ $\rightarrow$ **non-monotonic**.
   - `s_out_cnt` (Rank 5, \|SHAP\|=0.2546): Spearman $\rho = -0.0082$ $\rightarrow$ **non-monotonic**.
   - `r_gap_in` (Rank 6, \|SHAP\|=0.2195): Spearman $\rho = +0.6573$, Q1 = $-0.1419$, Q5 = $+0.5449$ $\rightarrow$ **higher raises score**.
   - `amount_ratio` (Rank 8, \|SHAP\|=0.1667): Spearman $\rho = +0.7025$, Q1 = $-0.3389$, Q5 = $+0.1035$ $\rightarrow$ **higher raises score**.
   - `r_in_amt` (Rank 9, \|SHAP\|=0.1281): Spearman $\rho = -0.7803$, Q1 = $+0.1914$, Q5 = $-0.1746$ $\rightarrow$ **higher lowers score**.

2. **Category Code Verification & Distribution**:
   - Replicated first-appearance dictionary over raw `HI-Small_Trans.csv`; verified exact row counts per code in `formatted_transactions.csv` (100% match).
   - **Payment Format**:
     - `Credit Card` (Code 2): Mean SHAP = **+3.0484**, represents **92.43%** of all test positives (vs 13.71% of negatives).
     - `Cheque` (Code 1): Mean SHAP = **-0.3205**, represents only **2.05%** of positives (vs 28.91% of negatives).
     - `Reinvestment` (Code 0): Mean SHAP = **+0.0030**, 3.72% of positives.
     - `ACH` (Code 3): Mean SHAP = **-0.8482**, 1.18% of positives.
     - `Cash` (Code 4): Mean SHAP = **-0.7504**, 0.00% of positives.

3. **New-Pair Full Test Set Check ($N = 863,900$)**:
   - `pair_prior_cnt == 0` (New Pair): Positive rate = **3.4282%** (1,276 / 37,221).
   - `pair_prior_cnt > 0` (Prior Pair): Positive rate = **0.0405%** (335 / 826,679).
   - Positives with `pair_prior_cnt == 0`: **79.21%** (1,276 of 1,611 test positives).
   - Negatives with `pair_prior_cnt == 0`: **4.17%** (35,945 of 862,289 test negatives).
   - **Enrichment Ratio: 84.60x** higher laundering rate on first-time pairings.

4. **Tenure Ablation (Dropping 8 Cumulative-Volume Counters)**:
   - Dropping `s_out_cnt`, `s_out_amt`, `s_in_cnt`, `s_in_amt`, `r_in_cnt`, `r_in_amt`, `r_out_cnt`, `r_out_amt` (31 $\rightarrow$ 23 features).
   - Full model: PR-AUC = `0.694059`, Lift = `372.2`, Unique scores = `862,501`.
   - Tenure-ablated model: PR-AUC = `0.694474`, Lift = `372.4`, Unique scores = `810,004`.
   - Result: PR-AUC improves slightly (**+0.000415**), confirming cumulative volume counters were non-essential and removing them eliminates potential time-proxy leakage.

5. **Contradictions to Prior Step 6 Summary**:
   - *Payment Format*: The previous summary guessed that *"high-risk formats like Wire or Cheque substantially elevate laundering likelihood"*. In reality, Cheque strongly lowers score (mean SHAP -0.3205; only 2.05% of positives), while **Credit Card** overwhelmingly dominates laundering cases (92.43% of positives, mean SHAP +3.0484).
   - *Sender Gap*: The previous summary asserted that *"very short gaps (rapid-fire transactions) increase laundering score"*. The empirical Spearman correlation is positive ($\rho = +0.2372$, Q1 = +0.3911 vs Q5 = +1.6730), showing that higher gap times actually produce higher SHAP values.
   - *Unique Receivers / Out Count*: Previously described as positive linear drivers; the data shows both are strictly **non-monotonic** ($\rho = -0.1701$ and $\rho = -0.0082$).

### Correction to Step 6b (Payment Format Category Mapping)

In the Step 6b summary above, **Table 2 (Payment Format Category Code Analysis)** suffered from a positional indexing mix-up: category names were resolved using `.cat.codes` positional index rather than the underlying integer code in the formatted dataset. Because code 0 (`Reinvestment`) was absent from the test split, the categories were shifted by 1 position (e.g., positional index 2 was looked up in the first-appearance dictionary as `Credit Card` instead of the actual data value 3, which is `ACH`).

The corrected payment-format SHAP table (evaluated across all payment format codes in the exact same SHAP sample of 21,611 transactions, with full test set row and positive counts) is:

| Code | Real Payment Format | Sample Count | Mean SHAP | Share of Sampled Positives (%) | Share of Sampled Negatives (%) | Full Test Rows | Full Test Positives |
|:----:|:-------------------|:------------:|:---------:|:------------------------------:|:------------------------------:|:--------------:|:-------------------:|
| 0 | Reinvestment | 0 | +0.0000 | 0.00% | 0.00% | 0 | 0 |
| 1 | Cheque | 8,167 | +0.0030 | 3.72% | 40.53% | 347,008 | 60 |
| 2 | Credit Card | 5,815 | -0.3205 | 2.05% | 28.91% | 250,247 | 33 |
| 3 | ACH | 4,232 | +3.0484 | 92.43% | 13.71% | 120,303 | 1,489 |
| 4 | Cash | 2,132 | -0.8482 | 1.18% | 10.56% | 91,737 | 19 |
| 5 | Wire | 681 | -0.7504 | 0.00% | 3.40% | 30,276 | 0 |
| 6 | Bitcoin | 584 | +1.0768 | 0.62% | 2.87% | 24,329 | 10 |

**Takeaway**: **ACH** (code 3) — not Credit Card — is the payment channel that overwhelmingly dominates laundering cases (92.43% of test positives, 1,489 of 1,611 test laundering rows, mean SHAP +3.0484). Credit Card (code 2) only accounts for 2.05% of positives (33 rows) and lowers the score (mean SHAP -0.3205).

---

## Step 7 — Report Assets, Visualizations, and Two Notebooks

### Goal
Produce the figures, tables, and two notebooks (
otebooks/01_eda.ipynb and 
otebooks/02_results.ipynb) required for the project report, modularized through a dedicated computation library (
eport_assets.py), reading results strictly from data files and persisted benchmark artifacts.

### Summary of New Files

| File | Status | Description |
|------|--------|-------------|
| 
eport_assets.py | **Created** | Reusable reporting module containing all data aggregation, table formatting, and matplotlib visualization functions (200 dpi, consistent model color palette, no seaborn). |
| 
otebooks/01_eda.ipynb | **Created** | Thin exploratory data analysis notebook displaying dataset overview, daily stream volume/rate, payment formats, currencies, transaction amounts, cyclical time patterns, and history dynamics. |
| 
otebooks/02_results.ipynb | **Created** | Thin results notebook presenting benchmark comparison tables across 6 experiments, lift across feature sets, split inflation factors, PR curves, alert budget curves, rule checks, tenure checks, and embedded SHAP interpretability. |
| create_notebooks.py | **Created** | Reproducible script generating both notebooks programmatically via 
bformat. |
| 
un_report_assets.py | **Created** | Standalone runner script executing all EDA and Results computations end-to-end and printing summary tables. |
| 
esults/final_tables.md | **Created** | Publication-ready Markdown tables of mean ± std across seeds for test PR-AUC, lift, ROC-AUC, F1, precision, and recall for all 6 experiments (excluding plain LightGBM). |
| 
esults/figures/report_eda_amounts.png | **Created** | Density-normalized histograms of log10(Amount Sent) for all positives vs. 200k sampled negatives (1800x1000 px). |
| 
esults/figures/report_eda_daily_laundering.png | **Created** | 2-panel chart of daily laundering rate (%) and total transaction volume across chronological days with split boundaries (2000x1400 px). |
| 
esults/figures/report_eda_hourly_weekly.png | **Created** | 2-panel chart of laundering rate across hours of day and days of week (2600x1000 px). |
| 
esults/figures/report_eda_pair_prior_cnt_daily.png | **Created** | Daily percentage of first-time pair transactions (pair_prior_cnt == 0) with train/val/test bounds (2000x1000 px). |
| 
esults/figures/report_eda_payment_format.png | **Created** | Grouped bar chart comparing share of all positives vs. negatives per payment format (2000x1000 px). |
| 
esults/figures/report_results_alert_budgets.png | **Created** | Alert-budget trade-off curves (Precision@k and Recall@k) at 0.1%, 0.5%, and 1.0% compliance budgets for RF, XGB, and LightGBM (2600x1000 px). |
| 
esults/figures/report_results_lift_by_features.png | **Created** | Grouped bar chart of PR-AUC lift per model across tabular, tab+dyn, and tab+dyn+static feature sets in 2 panels (temporal and random splits) with std error bars (2800x1200 px). |
| 
esults/figures/report_results_lift_inflation.png | **Created** | 2-panel visualization comparing temporal vs. random lift and quantifying inflation ratios (random / temporal lift) (2800x1200 px). |
| 
esults/figures/report_results_pr_curves.png | **Created** | High-resolution test Precision-Recall curves on temporal split for LightGBM (Undersample) comparing tabular baseline (PR-AUC 0.1860) to tab+dyn (PR-AUC 0.6941) against prevalence baseline (1600x1200 px). |

### Key Empirical Findings & Verified Numbers

1. **Dataset Overview (ormatted_transactions.csv)**:
   - Total rows: **5,078,345** | Distinct accounts: **515,088** | Days: **18** | Laundering rows: **5,177** | Prevalence: **0.1019%**.

2. **Payment Format & Category Code Clarification**:
   - True first-appearance dictionary verified against HI-Small_Trans.csv:
     - Code 0: Reinvestment (481,056 rows, 0 positives, 0.00% pos rate)
     - Code 1: Cheque (1,864,331 rows, 324 positives, 0.0174% pos rate, 6.26% pos share)
     - Code 2: Credit Card (1,323,324 rows, 206 positives, 0.0156% pos rate, 3.98% pos share)
     - Code 3: ACH (600,797 rows, 4,483 positives, 0.7462% pos rate, **86.59%** pos share across dataset, **92.43%** of test positives)
     - Code 4: Cash (490,891 rows, 108 positives, 0.0220% pos rate, 2.09% pos share)
     - Code 5: Wire (171,855 rows, 0 positives, 0.00% pos rate)
     - Code 6: Bitcoin (146,091 rows, 56 positives, 0.0383% pos rate, 1.08% pos share)
   - *Note on Step 6b Table*: In Step 6b, .cat.codes was applied to a sample containing 5 formats, shifting ACH (code 3) to internal position 2, which explain_checks matched to 'Credit Card'. In reality, **ACH (Code 3)** is the overwhelmingly dominant laundering channel.

3. **Pair Prior History & Enrichment**:
   - On the temporal test set (=863,900$, 1,611 positives):
     - First-time pairs (pair_prior_cnt == 0): 37,221 rows, 1,276 positives -> laundering rate = **3.4282%**.
     - Prior pairs (pair_prior_cnt > 0): 826,679 rows, 335 positives -> laundering rate = **0.0405%**.
     - **Enrichment Ratio: 84.60x** higher laundering likelihood on new directed pairings.

4. **Model Lift & Split Inflation**:
   - On temporal split, adding dynamic graph features dramatically expands lift for tree models:
     - Random Forest: 91.3x (tabular) -> **356.3x** (tab+dyn) -> 349.9x (tab+dyn+static).
     - XGBoost: 68.3x (tabular) -> **307.0x** (tab+dyn) -> 311.9x (tab+dyn+static).
     - LightGBM (Undersample): 92.4x (tabular) -> **379.7x** (tab+dyn) -> 367.8x (tab+dyn+static).
   - Random split inflation ratio ( / Temporal$ lift):
     - Tabular models: ~0.78x to 1.05x (no substantial inflation without graph features).
     - Dynamic graph models: **1.41x (RF), 1.67x (XGB), 1.49x (LGBM)** inflation under random partitioning, confirming that random cross-validation severely overstates performance due to bidirectional historical leakage.

5. **Inspection Alert Budgets (Temporal Tab+Dyn)**:
   - At 0.1% budget (864 alerts): LightGBM attains **92.28% precision** and captures **49.49% of all test positives**.
   - At 0.5% budget (4,320 alerts): LightGBM attains **30.15% precision** and captures **80.86% of all test positives**.
   - At 1.0% budget (8,639 alerts): LightGBM attains **16.17% precision** and captures **86.72% of all test positives**.

6. **Simple Heuristic Rule Check vs. Machine Learning**:
   - Literal Credit Card rule (Payment Format == Credit Card) & (pair_prior == 0): flags 14,748 rows, captures 0 positives (0.00% precision, 0.00% recall, 0.00 lift).
   - Dominant Channel rule (Payment Format == ACH) & (pair_prior == 0): flags 21,204 rows, captures 1,270 positives (**5.99% precision, 78.83% recall, 0.1113 F1, 32.12x lift**).
   - Machine Learning comparison at matching alert volumes:
     - LightGBM at Top-14,748: **9.61% precision, 87.96% recall, 0.1732 F1, 51.52x lift**.
     - LightGBM at Top-21,204: **6.75% precision, 88.83% recall, 0.1254 F1, 36.19x lift**.
     - Decision Tree (depth 3, balanced): learns (Payment Format == 3) & (pair_prior == 0), flags 229,161 rows (97.77% recall, 0.69% precision, test PR-AUC = 0.0302, 16.22x lift).
     - Full LightGBM continuous model: achieves **test PR-AUC = 0.6941 and 372.2x lift**.

---

## Step 7c — Result Literal Decoupling, Direction Table Persistence, and SHAP Verification

### Goal
Eliminate all typed result numbers from reporting code, persist `results/direction_table.csv` directly from `explain_checks.py`, replace hardcoded SHAP summary tables with dynamic reads from `results/category_shap_corrected.csv` and `results/direction_table.csv`, clean up the simple-rule comparison table, and re-execute both notebooks.

### Summary of Modifications

| File | Status | Description of Change | Rationale |
|---|---|---|---|
| `explain_checks.py` | **Modified** | Saved Table 1 results to `results/direction_table.csv` with columns `Rank,Feature,Mean |SHAP|,Spearman rho,Q1 Mean SHAP,Q5 Mean SHAP,Empirical Direction`. Updated Table 2 export to `results/category_shap_corrected.csv` with exact columns `format name,code,sampled rows,mean SHAP,share of sampled positives,share of sampled negatives,full-test rows,full-test positives`. | Ensures all interpretability tables are computed and saved directly by analysis routines without manual transcription. |
| `report_assets.py` | **Modified** | (1) Replaced hardcoded `df_dir_sample` and `df_cat_sample` in `results_shap_summary` with dynamic reads from `results/direction_table.csv` and `results/category_shap_corrected.csv`.<br>(2) In `results_simple_rule_check`, removed the Credit Card rule row and its matching top-N LGBM row; looked up ACH format code strictly by name (`"ACH"`) from verified map; dynamically computed Full LightGBM metrics from `results/results.csv`.<br>(3) In `results_alert_budget_table_and_plot`, dynamically generated budget tick labels from `te_budget_*_k`.<br>(4) In `eda_amounts_table_and_plot`, dynamically formatted positive count using `len(pos_amounts)`. | Adheres strictly to the Step 7c rule: no result number may be typed into reporting code. |
| `notebooks/01_eda.ipynb` | **Regenerated** | Re-executed in-place with `aml-venv` kernel. | Synchronizes outputs with decoupled reporting functions. |
| `notebooks/02_results.ipynb` | **Regenerated** | Re-executed in-place with `aml-venv` kernel. Displays updated Table 8a (direction table) and Table 8b (all 7 payment formats + top 8 currencies) without Credit Card misattribution. | Fully verified row-by-row against CSV artifacts; 0 instances of "Credit Card" near "3.0484" or "92.43". |
| `results/direction_table.csv` | **Created** | Persisted empirical direction table containing top 15 features by mean \|SHAP\|, Spearman rho, quintile means, and monotonic directions. | Direct artifact generated from `explain_checks.py`. |
| `results/category_shap_corrected.csv` | **Updated** | Standardized column names and content for all 7 payment formats and top 8 currencies. | Canonical source for category code SHAP distributions. |
| `verify_step7c.py` | **Created** | Verification script reading executed notebook JSON and asserting row-by-row equality with `results/direction_table.csv` and `results/category_shap_corrected.csv`, and checking string proximity. | Automated regression test for Step 7c integrity. |

### Key Empirical Verifications
- **Direction Table Row-by-Row**: Verified all 15 rows from `notebooks/02_results.ipynb` JSON match `results/direction_table.csv`.
- **Category Table Row-by-Row**: Verified all 15 rows (7 payment formats + 8 currencies) from `notebooks/02_results.ipynb` JSON match `results/category_shap_corrected.csv`.
- **Proximity Count**: Search for `"Credit Card"` within 100 characters of `"3.0484"` or `"92.43"` yielded **0** occurrences across both notebooks.

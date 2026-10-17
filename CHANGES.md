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


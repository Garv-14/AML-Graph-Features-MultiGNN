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

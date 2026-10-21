# AML Tabular Baselines — IBM HI-Small Study

A study of tabular and graph-feature machine-learning models for anti-money laundering (AML)
detection on the IBM HI-Small synthetic transaction dataset.
The project covers data formatting, chronological split export, tabular baselines
(Logistic Regression, Random Forest, XGBoost, LightGBM, Isolation Forest; plain class-weighted
LightGBM is run but excluded from reported tables, see Limitations), leakage-safe dynamic and static
graph features, a three-way ablation (tabular / tab+dyn / tab+dyn+static), SHAP
interpretability, and report-ready figures and notebooks.

**Built on IBM/Multi-GNN (Apache-2.0)** —
[IBM/Multi-GNN](https://github.com/IBM/Multi-GNN).
The following components are inherited from that repository without modification to their
model logic: the Kaggle data formatter (`format_kaggle_files.py`), the temporal split logic
inside `data_loading.py` and `train_util.py`, and the GNN model classes in `models.py`
(`GINe`, `GATe`, `PNA`, `RGCN`).

**What is new in this repository:**
- `evaluate.py` — shared, model-agnostic evaluation module (PR-AUC, ROC-AUC, F1, alert-budget metrics)
- `run_tabular.py`, `features_tabular.py` — tabular baseline pipeline (6 model families, 2 split kinds)
- `graph_features.py` — vectorised leakage-safe dynamic and static graph features
- `summarize_ablation.py`, `summarize_results.py` — ablation across three feature sets
- `explain_shap.py`, `explain_checks.py` — SHAP analysis and direction/category verification
- `report_assets.py`, `run_report_assets.py`, `create_notebooks.py` — figures and notebook generation
- `results/` — persisted CSV results, Markdown tables, and figures

**Authors:** Garv Chhamalwan · **Supervisor:** [SUPERVISOR]

---

## Dataset

**IBM Transactions for Anti-Money Laundering (HI-Small)**
Kaggle dataset: [ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

The data is **not included** in this repository. You must download it separately under its
own terms (see the Kaggle dataset page). After downloading, place the raw file at:

```
<AML_DATA_DIR>/Small_HI/HI-Small_Trans.csv
```

The data directory is configured by the environment variable `AML_DATA_DIR` (default
`C:/aml_data`) or by editing the `aml_data` path in `data_config.json`.
See [SETUP.md](SETUP.md) for full instructions.

---

## Reproduction Guide

> **Note:** wandb logging is disabled by default. Pass `--use_wandb` to `main.py` to enable it.

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

(or `requirements.txt` plus PyTorch CPU and `torch_geometric` as in [SETUP.md](SETUP.md)). PyTorch is needed even for the tabular pipeline because the split code imports it.

### 2. Run unit tests

```bash
python tests/test_evaluate.py
python tests/test_graph_features.py
```

### 3. Format raw Kaggle transactions

```bash
python format_kaggle_files.py "<AML_DATA_DIR>\Small_HI\HI-Small_Trans.csv"
```

Produces `<AML_DATA_DIR>/Small_HI/formatted_transactions.csv` sorted by `Timestamp`.

### 4. Export the temporal split

```bash
python export_splits.py --data Small_HI
```

Writes `splits/Small_HI_split.npz` and `splits/Small_HI_split_summary.json`.

### 5. Export the random split

```bash
python make_random_split.py
```

Writes `splits/Small_HI_random_split.npz` and `splits/Small_HI_random_split_summary.json`.

### 6. Compute graph features

```bash
python graph_features.py
```

Builds `graph_dynamic.parquet` and the two static files (`graph_static_temporal.parquet` and `graph_static_random.parquet`) in `<AML_DATA_DIR>/Small_HI/`. Requires both temporal and random splits to have been exported. Dynamic features take roughly 24 s; static features roughly 5 s per split.

### 7. Run tabular baselines (both split kinds)

```bash
python run_tabular.py --split_kind temporal --seeds 0 1 2
python run_tabular.py --split_kind random   --seeds 0 1 2
```

Appends rows to `results/results.csv`. The default model list also trains plain LightGBM and Isolation Forest; plain LightGBM rows stay in `results/results.csv` but are excluded from reported tables.

### 8. Summarize tabular baseline results

```bash
python summarize_results.py
```

Aggregates baseline rows from `results/results.csv` into `results/summary_tabular.md`.

### 9. Run ablation over feature sets

```bash
python run_tabular.py --feature_set tab_dyn        --split_kind temporal --seeds 0 1 2
python run_tabular.py --feature_set tab_dyn        --split_kind random   --seeds 0 1 2
python run_tabular.py --feature_set tab_dyn_static --split_kind temporal --seeds 0 1 2
python run_tabular.py --feature_set tab_dyn_static --split_kind random   --seeds 0 1 2
```

### 10. Summarize ablation results

```bash
python summarize_ablation.py
```

Aggregates feature set ablation rows from `results/results.csv` into `results/summary_ablation.md`.

### 11. Run SHAP interpretability

```bash
python explain_shap.py
```

Retrains the best model (`lightgbm_undersample`, `tab_dyn`, temporal, seed 0), verifies
PR-AUC against `results/results.csv`, and saves figures to `results/figures/`.

### 12. Run tenure and category checks

```bash
python explain_checks.py
```

Writes `results/direction_table.csv` and `results/category_shap_corrected.csv`.

### 13. Generate report assets

```bash
python run_report_assets.py
```

Produces all EDA and results figures in `results/figures/` and Markdown tables.

### 14. Register Jupyter kernel

```bash
python -m ipykernel install --user --name aml-venv --display-name "AML venv"
```

### 15. Generate and execute notebooks

```bash
python create_notebooks.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=aml-venv --ExecutePreprocessor.timeout=1800 notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=aml-venv --ExecutePreprocessor.timeout=1800 notebooks/02_results.ipynb
```

Creates `notebooks/01_eda.ipynb` and `notebooks/02_results.ipynb` and executes them in-place with the `aml-venv` kernel.

### 16. Update results table in this README

```bash
python update_readme_results.py
```

Reads `results/results.csv` and writes the lift table between the markers below.

---

## Repository Layout

```
.gitignore                      # excludes .venv, models/, results/preds/, splits/*.npz, etc.
CHANGES.md                      # detailed per-step change log
LICENSE                         # Apache-2.0 (inherited)
README.md                       # this file
README_IBM_MultiGNN.md          # original IBM/Multi-GNN README (kept for attribution)
SETUP.md                        # installation and dataset setup guide
requirements.txt                # direct dependencies
requirements-lock.txt           # pinned pip freeze snapshot

# ── Inherited GNN pipeline ────────────────────────────────────────────────
main.py                         # GNN training entry point (inherited)
models.py                       # GINe, GATe, PNA, RGCN model classes (inherited)
training.py                     # GNN training loop (inherited, wandb disabled by default)
train_util.py                   # PyG LinkNeighborLoader setup, AddEgoIds transform, GNN evaluation (evaluate_homo/evaluate_hetero), checkpointing, and model_settings.json lookup (inherited)
inference.py                    # GNN inference (inherited; known broken, see CHANGES.md)
data_loading.py                 # temporal split logic (inherited, env-var path added)
data_util.py                    # graph data utilities (inherited)
util.py                         # shared CLI args (--use_wandb flag added)
format_kaggle_files.py          # Kaggle CSV formatter (inherited; datatable replaced)
env.yml                         # Conda environment (inherited; pip path preferred)
model_settings.json             # GNN hyperparameter presets (inherited)

# ── New: data access and shared evaluation ────────────────────────────────
data_access.py                  # central load_formatted / load_split helpers
data_config.json                # configures AML_DATA_DIR and model save paths
evaluate.py                     # PR-AUC, ROC-AUC, F1, alert-budget metrics, append_result
export_splits.py                # exports temporal split indices and summary JSON
make_random_split.py            # exports random split with matching partition sizes
sanity_check.py                 # verifies split disjointness and laundering rates

# ── New: tabular baseline pipeline ────────────────────────────────────────
features_tabular.py             # 11 per-row tabular features (no cross-row aggregation)
run_tabular.py                  # trains 6 models, 3 feature sets, 2 splits, N seeds
summarize_results.py            # aggregates results.csv → results/summary_tabular.md
summarize_ablation.py           # aggregates ablation rows → results/summary_ablation.md

# ── New: graph features ────────────────────────────────────────────────────
graph_features.py               # vectorised dynamic + static graph feature builder

# ── New: SHAP interpretability ────────────────────────────────────────────
explain_shap.py                 # SHAP TreeExplainer on best model; saves figures
explain_checks.py               # direction analysis, category decoding, tenure ablation

# ── New: report assets and notebooks ──────────────────────────────────────
report_assets.py                # all plotting and table functions (no typed results)
run_report_assets.py            # runs all EDA + results computations end-to-end
create_notebooks.py             # generates notebooks/01_eda.ipynb and 02_results.ipynb
update_readme_results.py        # writes lift table into README.md from CSV
verify_step7c.py                # regression test: notebook outputs match CSV artifacts

# ── Results and figures ───────────────────────────────────────────────────
results/results.csv             # all experiment rows (every seed, split, feature set)
results/results_v1_backup.csv   # backup CSV from earlier run v1
results/results_v2_backup.csv   # backup CSV from earlier run v2
results/results_v3_backup.csv   # backup CSV from earlier run v3
results/tenure_check.csv        # full vs. tenure-ablated model comparison (1 seed)
results/final_tables.md         # publication-ready mean ± std tables
results/summary_tabular.md      # tabular baseline summary
results/summary_ablation.md     # ablation summary
results/direction_table.csv     # SHAP direction analysis (Spearman rho, quintile means)
results/category_shap_corrected.csv  # corrected payment-format SHAP distributions
results/figures/                # PNG figures (EDA, results, SHAP)

# ── Splits ────────────────────────────────────────────────────────────────
splits/Small_HI_split_summary.json         # temporal split record counts and boundaries
splits/Small_HI_random_split_summary.json  # random split record counts and boundaries
splits/*.npz                               # index arrays (git-ignored; regenerate locally)

# ── Notebooks ─────────────────────────────────────────────────────────────
notebooks/01_eda.ipynb          # exploratory data analysis
notebooks/02_results.ipynb      # benchmark results and interpretability

# ── Tests ─────────────────────────────────────────────────────────────────
tests/test_evaluate.py          # unit tests for evaluate.py (metric math)
tests/test_graph_features.py    # unit tests for graph_features.py (leakage, coverage)
```

---

## Results

<!-- RESULTS:START -->

Test PR-AUC **lift** (= PR-AUC / test prevalence), mean ± std over seeds.

| Feature set | Split | Model | Seeds | Lift (mean ± std) |
|:------------|:------|:------|------:|------------------:|
| tabular | temporal | Random Forest | 3 | 91.3 ± 5.2 |
| tabular | temporal | XGBoost | 3 | 68.3 ± 4.4 |
| tabular | temporal | LightGBM (Undersample) | 3 | 92.4 ± 7.6 |
| tabular | random | Random Forest | 3 | 71.4 ± 1.1 |
| tabular | random | XGBoost | 3 | 71.9 ± 4.7 |
| tabular | random | LightGBM (Undersample) | 3 | 80.8 ± 1.9 |
| tab_dyn | temporal | Random Forest | 3 | 356.3 ± 4.0 |
| tab_dyn | temporal | XGBoost | 3 | 307.0 ± 62.6 |
| tab_dyn | temporal | LightGBM (Undersample) | 3 | 379.7 ± 6.9 |
| tab_dyn | random | Random Forest | 3 | 500.8 ± 3.3 |
| tab_dyn | random | XGBoost | 3 | 512.7 ± 6.9 |
| tab_dyn | random | LightGBM (Undersample) | 3 | 565.7 ± 2.7 |
| tab_dyn_static | temporal | Random Forest | 3 | 349.9 ± 1.2 |
| tab_dyn_static | temporal | XGBoost | 3 | 311.9 ± 8.8 |
| tab_dyn_static | temporal | LightGBM (Undersample) | 3 | 367.8 ± 15.5 |
| tab_dyn_static | random | Random Forest | 3 | 512.5 ± 2.2 |
| tab_dyn_static | random | XGBoost | 3 | 515.2 ± 4.8 |
| tab_dyn_static | random | LightGBM (Undersample) | 3 | 577.0 ± 1.2 |

Test prevalence: temporal split = 0.1865%, random split = 0.1071%. All experiments run over 3 seeds (0, 1, 2).
<!-- RESULTS:END -->

---

## Limitations

- Only one dataset (HI-Small) is evaluated; it is synthetic, so no claim is made about real-world AML detection performance.
- The three seeds vary the model's randomness, not the data split.
- The test set has only on the order of a thousand positives, so small differences between models are within noise.
- XGBoost varied noticeably between seeds.
- Account-history features are shorter for training rows than for test rows because history accumulates over time.
- Raw PR-AUC is not comparable between the temporal and random splits because prevalence differs (use lift).
- Plain LightGBM: with scale_pos_weight = negatives/positives, training stopped after one tree (validation average precision peaked at the first round). The cause was not fully diagnosed. It is excluded from reported tables.
- No GNN results were produced. The inherited Multi-GNN code is kept, but its neighbour sampler needs pyg-lib or torch-sparse, which were not installed, and GNN training was not pursued on the Windows/Python 3.13 setup used here. The GNN pipeline is therefore untested.
- Reading the inherited code showed that validation and test edge features are normalised with their own statistics (not training statistics), that the node set is built from all rows, and that neighbour sampling has no time constraint. This comes from code inspection, not experiments, and matters only if the GNN is used.
- On the temporal split static PageRank/degree features did not help; on the random split they gave a small gain. The reason was not investigated.
- The finding that ACH carries most laundering rows is a property of this particular synthetic dataset; it should not be generalised.
- A gap between random-split and temporal-split performance is measured, but its cause (distribution shift, bidirectional history leakage in random split, or both) was not tested with controlled experiments.
- SHAP values explain what the trained model relies on, not true risk factors. The SHAP computation was run on all test positives plus a 20 k subsample of negatives, so only feature rankings carry over; magnitudes do not generalise to the full test distribution.
- The tenure check used one seed. With only a few thousand positives, small differences between the full and tenure-ablated models are within noise.
- NetworkX is used only to validate the PageRank implementation in unit tests; the feature computation itself uses vectorised pandas/NumPy with `scipy.sparse`.

---

## Citations

The following citations are taken from [README_IBM_MultiGNN.md](README_IBM_MultiGNN.md):

1. Egressy, B., von Niederhäusern, L., Blanuša, J., Altman, E., Wattenhofer, R., & Atasu, K. (2024). *Provably Powerful Graph Neural Networks for Directed Multigraphs.* AAAI 2024. arXiv:2306.11586.

2. Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., & Atasu, K. (2023). *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023. arXiv:2306.16424. *(author list: verify against arXiv before submission)*

3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural
   Networks?* ICLR 2019. arXiv:1810.00826. *(GIN)*

4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).
   *Graph Attention Networks.* ICLR 2018. arXiv:1710.10903. *(GAT)*

5. Corso, G., Cavalleri, L., Beaini, D., Liò, P., & Veličković, P. (2020).
   *Principal Neighbourhood Aggregation for Graph Nets.* NeurIPS 2020. arXiv:2004.05718.
   *(PNA)*

6. Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M.
   (2018). *Modeling Relational Data with Graph Convolutional Networks.* ESWC 2018.
   arXiv:1703.06103. *(RGCN)*

7. Blanuša, J., Cravero Baraja, M., Anghel, A., von Niederhäusern, L., Altman, E., Pozidis, H., & Atasu, K. (2024). *Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection.* arXiv:2402.08593. *(venue: verify before submission)*

Author lists were checked against arXiv for the first two entries above; check all entries before submission.

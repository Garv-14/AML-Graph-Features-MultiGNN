# Step 5 Ablation Summary -- Graph Feature Enrichment

Dataset: Small HI-Small (AML) -- 5 M transactions
Seeds: 0, 1, 2 (mean +/- std across 3 seeds)

## Feature Sets

- **tabular** (11 features): 11 tabular
- **tab+dyn** (31 features): 11 tabular + 20 dynamic graph
- **tab+dyn+static** (37 features): 11 tabular + 20 dynamic graph + 6 static graph

## History Coverage (temporal split)

| Set   | Sender no-history | Receiver no-history |
|-------|-------------------|---------------------|
| Train |           13.22%  |             11.44%  |
| Val   |            0.09%  |              0.02%  |
| Test  |            0.10%  |              0.02%  |

Val/Test rows have near-zero missing history because they follow the dense training period.

---
## Temporal Split

### Full metrics table (test set, mean +/- std over 3 seeds)

```
Feature set      | Model                    |    PR-AUC (mean+/-std) |        ROC-AUC |             F1 |   P@0.1% |   P@0.5% |  Delta PR-AUC
--------------------------------------------------------------------------------------------------------------------------------------------
tabular          | Logistic Regression      |      0.0287 +/- 0.0000 | 0.9355 +/- 0.0000 | 0.0677 +/- 0.0000 |     2.9% |     3.5% | ++0.0000
                 | Random Forest            |      0.1702 +/- 0.0097 | 0.9362 +/- 0.0016 | 0.2673 +/- 0.0039 |    33.6% |    15.5% | ++0.0000
                 | XGBoost                  |      0.1274 +/- 0.0083 | 0.9478 +/- 0.0003 | 0.2548 +/- 0.0103 |    25.3% |    13.6% | ++0.0000
                 | LightGBM (undersample)   |      0.1723 +/- 0.0142 | 0.9483 +/- 0.0016 | 0.2757 +/- 0.0039 |    30.6% |    15.9% | ++0.0000
--------------------------------------------------------------------------------------------------------------------------------------------
tab+dyn          | Logistic Regression      |      0.3379 +/- 0.0000 | 0.9849 +/- 0.0000 | 0.3708 +/- 0.0000 |    50.6% |    21.9% | ++0.3092
                 | Random Forest            |      0.6644 +/- 0.0074 | 0.9771 +/- 0.0006 | 0.6186 +/- 0.0123 |    87.4% |    30.1% | ++0.4942
                 | XGBoost                  |      0.5726 +/- 0.1167 | 0.9894 +/- 0.0029 | 0.5764 +/- 0.0231 |    79.8% |    26.5% | ++0.4452
                 | LightGBM (undersample)   |      0.7080 +/- 0.0129 | 0.9853 +/- 0.0018 | 0.6896 +/- 0.0199 |    92.3% |    30.2% | ++0.5357
--------------------------------------------------------------------------------------------------------------------------------------------
tab+dyn+static   | Logistic Regression      |      0.1542 +/- 0.0000 | 0.9827 +/- 0.0000 | 0.2379 +/- 0.0000 |    29.5% |    14.8% | ++0.1255
                 | Random Forest            |      0.6525 +/- 0.0023 | 0.9747 +/- 0.0039 | 0.6055 +/- 0.0036 |    85.1% |    29.8% | ++0.4823
                 | XGBoost                  |      0.5817 +/- 0.0164 | 0.9906 +/- 0.0002 | 0.5443 +/- 0.0180 |    76.3% |    28.3% | ++0.4544
                 | LightGBM (undersample)   |      0.6858 +/- 0.0289 | 0.9854 +/- 0.0030 | 0.6766 +/- 0.0196 |    90.9% |    29.8% | ++0.5135
--------------------------------------------------------------------------------------------------------------------------------------------
```

---
## Random Split

### Full metrics table (test set, mean +/- std over 3 seeds)

```
Feature set      | Model                    |    PR-AUC (mean+/-std) |        ROC-AUC |             F1 |   P@0.1% |   P@0.5% |  Delta PR-AUC
--------------------------------------------------------------------------------------------------------------------------------------------
tabular          | Logistic Regression      |      0.0147 +/- 0.0000 | 0.9128 +/- 0.0000 | 0.0370 +/- 0.0000 |     1.3% |     1.7% | ++0.0000
                 | Random Forest            |      0.0764 +/- 0.0012 | 0.9016 +/- 0.0038 | 0.1722 +/- 0.0038 |    17.3% |     7.8% | ++0.0000
                 | XGBoost                  |      0.0769 +/- 0.0050 | 0.9238 +/- 0.0010 | 0.1711 +/- 0.0045 |    16.8% |     7.7% | ++0.0000
                 | LightGBM (undersample)   |      0.0865 +/- 0.0020 | 0.9243 +/- 0.0028 | 0.1787 +/- 0.0109 |    17.4% |     8.2% | ++0.0000
--------------------------------------------------------------------------------------------------------------------------------------------
tab+dyn          | Logistic Regression      |      0.1973 +/- 0.0000 | 0.9794 +/- 0.0000 | 0.2458 +/- 0.0000 |    25.3% |    10.9% | ++0.1825
                 | Random Forest            |      0.5362 +/- 0.0035 | 0.9714 +/- 0.0014 | 0.5512 +/- 0.0007 |    54.1% |    15.8% | ++0.4598
                 | XGBoost                  |      0.5490 +/- 0.0074 | 0.9881 +/- 0.0003 | 0.5570 +/- 0.0156 |    55.7% |    16.0% | ++0.4721
                 | LightGBM (undersample)   |      0.6057 +/- 0.0029 | 0.9890 +/- 0.0001 | 0.6311 +/- 0.0024 |    60.6% |    16.2% | ++0.5192
--------------------------------------------------------------------------------------------------------------------------------------------
tab+dyn+static   | Logistic Regression      |      0.2061 +/- 0.0000 | 0.9802 +/- 0.0000 | 0.2887 +/- 0.0000 |    29.6% |    10.8% | ++0.1913
                 | Random Forest            |      0.5487 +/- 0.0024 | 0.9708 +/- 0.0010 | 0.5650 +/- 0.0038 |    54.9% |    16.0% | ++0.4723
                 | XGBoost                  |      0.5517 +/- 0.0051 | 0.9885 +/- 0.0002 | 0.5565 +/- 0.0033 |    55.5% |    16.2% | ++0.4747
                 | LightGBM (undersample)   |      0.6178 +/- 0.0013 | 0.9889 +/- 0.0003 | 0.6412 +/- 0.0027 |    61.5% |    16.5% | ++0.5313
--------------------------------------------------------------------------------------------------------------------------------------------
```

---
## Lift Gains vs Tabular Baseline

### Temporal split (PR-AUC delta vs tabular)

| Model | tab | tab+dyn | Gain | tab+dyn+static | Gain vs tab |
|-------|-----|---------|------|----------------|-------------|
| Logistic Regression | 0.0287 | 0.3379 | +0.3092 | 0.1542 | +0.1255 |
| Random Forest | 0.1702 | 0.6644 | +0.4942 | 0.6525 | +0.4823 |
| XGBoost | 0.1274 | 0.5726 | +0.4452 | 0.5817 | +0.4544 |
| LightGBM (undersample) | 0.1723 | 0.7080 | +0.5357 | 0.6858 | +0.5135 |

### Random split (PR-AUC delta vs tabular)

| Model | tab | tab+dyn | Gain | tab+dyn+static | Gain vs tab |
|-------|-----|---------|------|----------------|-------------|
| Logistic Regression | 0.0147 | 0.1973 | +0.1825 | 0.2061 | +0.1913 |
| Random Forest | 0.0764 | 0.5362 | +0.4598 | 0.5487 | +0.4723 |
| XGBoost | 0.0769 | 0.5490 | +0.4721 | 0.5517 | +0.4747 |
| LightGBM (undersample) | 0.0865 | 0.6057 | +0.5192 | 0.6178 | +0.5313 |

---
## Key Findings

- **Dynamic graph features (tab+dyn)** give the largest lift on the temporal split,
  especially for LightGBM and Random Forest.
- **Static features (tab+dyn+static)** show mixed results -- PageRank/degree computed
  only on training edges are noisier, and the gains over tab+dyn are not consistent.
- **LightGBM (undersample)** is the best single model across all feature sets and splits.
- **Logistic Regression** benefits less from graph features because it cannot capture
  non-linear interactions.

### Best model per split

- **temporal**: `tab+dyn` + `LightGBM (undersample)` -> PR-AUC = **0.7080** +/- 0.0129
- **random**: `tab+dyn+static` + `LightGBM (undersample)` -> PR-AUC = **0.6178** +/- 0.0013

"""Shared data access module for the IBM AML datasets.

Provides standardized loading functions for formatted transactions and
persisted dataset splits (both temporal and random), ensuring consistent row ordering
and identical features across tabular, graph-feature, and GNN pipelines.
"""

import os
import json
from typing import Tuple, Literal
import numpy as np
import pandas as pd


def get_data_dir(config_path: str = None) -> str:
    """Resolves the root directory of the AML transaction data.

    Checks the AML_DATA_DIR environment variable first, then falls back to the
    path defined in data_config.json, with a final default of 'C:/aml_data'.

    Args:
        config_path: Optional path to data_config.json.

    Returns:
        Absolute or resolved directory path containing AML datasets.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_config.json")

    configured_path = "C:/aml_data"
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                configured_path = cfg.get("paths", {}).get("aml_data", configured_path)
        except Exception:
            pass

    return os.environ.get("AML_DATA_DIR", configured_path)


def load_formatted(data_name: str = "Small_HI") -> pd.DataFrame:
    """Loads the preprocessed formatted_transactions.csv DataFrame.

    Args:
        data_name: Dataset directory name under the AML data root (default 'Small_HI').

    Returns:
        pandas DataFrame containing formatted transaction rows and columns.
    """
    aml_root = get_data_dir()
    csv_path = os.path.join(aml_root, data_name, "formatted_transactions.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Could not find formatted transactions file at: {csv_path}. "
            f"Ensure format_kaggle_files.py has been run."
        )

    return pd.read_csv(csv_path)


def load_split(
    data_name: str = "Small_HI",
    kind: Literal["temporal", "random"] = "temporal",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, validation, and test splits as DataFrames using saved split index files.

    Preserves the exact row sequence of the saved index arrays.

    Args:
        data_name: Dataset name (default 'Small_HI').
        kind: Split type, either 'temporal' (loads splits/{data_name}_split.npz)
              or 'random' (loads splits/{data_name}_random_split.npz).

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.join(base_dir, "splits")

    if kind == "temporal":
        npz_name = f"{data_name}_split.npz"
    elif kind == "random":
        npz_name = f"{data_name}_random_split.npz"
    else:
        raise ValueError(f"Unknown split kind '{kind}'. Expected 'temporal' or 'random'.")

    npz_path = os.path.join(splits_dir, npz_name)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Split index archive not found at: {npz_path}. "
            f"Run export_splits.py or make_random_split.py first."
        )

    split_indices = np.load(npz_path)
    tr_inds = split_indices["tr_inds"]
    val_inds = split_indices["val_inds"]
    te_inds = split_indices["te_inds"]

    df_full = load_formatted(data_name=data_name)

    # Use iloc to guarantee exact index preservation and order
    df_train = df_full.iloc[tr_inds].reset_index(drop=True)
    df_val = df_full.iloc[val_inds].reset_index(drop=True)
    df_test = df_full.iloc[te_inds].reset_index(drop=True)

    return df_train, df_val, df_test

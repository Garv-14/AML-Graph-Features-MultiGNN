"""Sanity check script for data_access.py and persisted dataset splits.

Verifies:
1. Loading formatted transactions and splits (temporal and random).
2. Exact row count preservation across splits.
3. Zero index leakage or overlap between train, validation, and test sets.
4. Laundering rates and distributions across splits.
"""

import os
import numpy as np
from tabulate import tabulate
from data_access import load_split, get_data_dir


def run_sanity_check(data_name: str = "Small_HI"):
    print("=" * 85)
    print(f"RUNNING SANITY CHECK FOR DATASET: {data_name}")
    print(f"Data root directory: {get_data_dir()}")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.join(base_dir, "splits")

    # 1. Check Temporal Split
    print("\n--- 1. Checking Temporal Split ---")
    temp_npz = os.path.join(splits_dir, f"{data_name}_split.npz")
    assert os.path.isfile(temp_npz), f"Missing temporal split file: {temp_npz}"
    temp_data = np.load(temp_npz)
    tr_temp = temp_data["tr_inds"]
    val_temp = temp_data["val_inds"]
    te_temp = temp_data["te_inds"]

    # Verify no overlap in temporal split
    assert len(set(tr_temp) & set(val_temp)) == 0, "ERROR: Overlap between temporal train and val!"
    assert len(set(tr_temp) & set(te_temp)) == 0, "ERROR: Overlap between temporal train and test!"
    assert len(set(val_temp) & set(te_temp)) == 0, "ERROR: Overlap between temporal val and test!"
    print("OK: Temporal split indices are strictly disjoint (0 overlap).")

    df_tr_t, df_val_t, df_te_t = load_split(data_name=data_name, kind="temporal")
    assert len(df_tr_t) == len(tr_temp), "Temporal train row count mismatch!"
    assert len(df_val_t) == len(val_temp), "Temporal val row count mismatch!"
    assert len(df_te_t) == len(te_temp), "Temporal test row count mismatch!"
    print("OK: DataFrames loaded via data_access.load_split(kind='temporal') match index lengths.")

    # 2. Check Random Split
    print("\n--- 2. Checking Random Split ---")
    rand_npz = os.path.join(splits_dir, f"{data_name}_random_split.npz")
    assert os.path.isfile(rand_npz), f"Missing random split file: {rand_npz}"
    rand_data = np.load(rand_npz)
    tr_rand = rand_data["tr_inds"]
    val_rand = rand_data["val_inds"]
    te_rand = rand_data["te_inds"]

    # Verify no overlap in random split
    assert len(set(tr_rand) & set(val_rand)) == 0, "ERROR: Overlap between random train and val!"
    assert len(set(tr_rand) & set(te_rand)) == 0, "ERROR: Overlap between random train and test!"
    assert len(set(val_rand) & set(te_rand)) == 0, "ERROR: Overlap between random val and test!"
    print("OK: Random split indices are strictly disjoint (0 overlap).")

    df_tr_r, df_val_r, df_te_r = load_split(data_name=data_name, kind="random")
    assert len(df_tr_r) == len(tr_rand), "Random train row count mismatch!"
    assert len(df_val_r) == len(val_rand), "Random val row count mismatch!"
    assert len(df_te_r) == len(te_rand), "Random test row count mismatch!"
    print("OK: DataFrames loaded via data_access.load_split(kind='random') match index lengths.")

    # 3. Comparative Summary Table
    def row_info(split_name, df):
        n = len(df)
        pos = int(df["Is Laundering"].sum())
        rate = (pos / n * 100.0) if n > 0 else 0.0
        return [f"{n:,}", f"{pos:,}", f"{rate:.4f}%"]

    table_data = [
        ["Temporal Train"] + row_info("Temporal Train", df_tr_t),
        ["Temporal Validation"] + row_info("Temporal Val", df_val_t),
        ["Temporal Test"] + row_info("Temporal Test", df_te_t),
        ["---", "---", "---", "---"],
        ["Random Train"] + row_info("Random Train", df_tr_r),
        ["Random Validation"] + row_info("Random Val", df_val_r),
        ["Random Test"] + row_info("Random Test", df_te_r),
    ]

    headers = ["Split", "Row Count", "Laundering Count", "Laundering Rate"]
    print("\n" + "=" * 85)
    print("COMPARATIVE SPLIT SUMMARY (Temporal vs. Random)")
    print("=" * 85)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 85)
    print("\nSANITY CHECK PASSED: All indices disjoint, non-overlapping, and verified.")


if __name__ == "__main__":
    run_sanity_check()

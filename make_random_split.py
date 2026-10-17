"""Generates a random train/val/test split matching the temporal split partition sizes.

Enables the "Temporal vs. Random Split" benchmark experiment to investigate
future look-ahead bias and distribution drift in AML detection.
"""

import os
import json
import numpy as np
from tabulate import tabulate
from data_access import load_formatted


def make_random_split(
    data_name: str = "Small_HI",
    seed: int = 42,
    n_train: int = 3248921,
    n_val: int = 965524,
    n_test: int = 863900,
):
    """Generates and persists a random split with identical partition counts to the temporal split."""
    print(f"Loading formatted transactions for {data_name}...")
    df = load_formatted(data_name=data_name)
    total_rows = len(df)
    expected_total = n_train + n_val + n_test

    if total_rows != expected_total:
        raise ValueError(
            f"Expected {expected_total} rows, but formatted dataset contains {total_rows} rows."
        )

    print(f"Generating random permutation with seed {seed}...")
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(total_rows)

    tr_inds = shuffled_indices[:n_train]
    val_inds = shuffled_indices[n_train : n_train + n_val]
    te_inds = shuffled_indices[n_train + n_val :]

    assert len(tr_inds) == n_train
    assert len(val_inds) == n_val
    assert len(te_inds) == n_test
    # Verify no overlap
    assert len(set(tr_inds).intersection(set(val_inds))) == 0
    assert len(set(tr_inds).intersection(set(te_inds))) == 0
    assert len(set(val_inds).intersection(set(te_inds))) == 0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    npz_path = os.path.join(splits_dir, f"{data_name}_random_split.npz")
    np.savez_compressed(npz_path, tr_inds=tr_inds, val_inds=val_inds, te_inds=te_inds)
    print(f"Saved random split indices to: {npz_path}")

    # Compute laundering rates
    y = df["Is Laundering"].to_numpy().astype(int)
    timestamps = df["Timestamp"].to_numpy()

    def get_split_stats(indices, name):
        count = len(indices)
        laundering = int(np.sum(y[indices]))
        pct = (laundering / count * 100.0) if count > 0 else 0.0
        ts_min = float(np.min(timestamps[indices])) if count > 0 else 0.0
        ts_max = float(np.max(timestamps[indices])) if count > 0 else 0.0
        return {
            "split": name,
            "num_rows": count,
            "pct_total_rows": round(count / total_rows * 100.0, 2),
            "num_laundering": laundering,
            "pct_laundering": round(pct, 4),
            "timestamp_min": round(ts_min, 1),
            "timestamp_max": round(ts_max, 1),
            "duration_hours": round((ts_max - ts_min) / 3600.0, 2),
        }

    train_stats = get_split_stats(tr_inds, "Train")
    val_stats = get_split_stats(val_inds, "Validation")
    test_stats = get_split_stats(te_inds, "Test")

    summary = {
        "dataset": data_name,
        "split_type": "random",
        "random_seed": seed,
        "total_transactions": total_rows,
        "total_laundering": int(np.sum(y)),
        "total_pct_laundering": round(float(np.sum(y)) / total_rows * 100.0, 4),
        "splits": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        },
    }

    json_path = os.path.join(splits_dir, f"{data_name}_random_split_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved random split summary to: {json_path}")

    # Print Table
    table_rows = [
        [
            s["split"],
            f"{s['num_rows']:,}",
            f"{s['pct_total_rows']}%",
            f"{s['num_laundering']:,}",
            f"{s['pct_laundering']:.3f}%",
            f"{s['timestamp_min']:.0f}s - {s['timestamp_max']:.0f}s",
            f"{s['duration_hours']:.1f}h",
        ]
        for s in [train_stats, val_stats, test_stats]
    ]

    headers = ["Split", "Rows", "% of Total", "Laundering Rows", "Laundering %", "Timestamp Range", "Duration"]
    print("\n" + "=" * 85)
    print(f"RANDOM SPLIT SUMMARY: {data_name} (Fixed Seed = {seed})")
    print("=" * 85)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))
    print("=" * 85 + "\n")


if __name__ == "__main__":
    make_random_split()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import itertools
import argparse
import numpy as np
import torch
from util import create_parser, set_seed
from data_loading import get_data
from tabulate import tabulate

def export_splits():
    parser = create_parser()
    for action in parser._actions:
        if action.dest in ['model', 'data']:
            action.required = False
    parser.set_defaults(data='Small_HI', model='gin')
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_config.json')
    with open(config_path, 'r') as config_file:
        data_config = json.load(config_file)

    set_seed(args.seed)

    print(f"Loading data for dataset: {args.data}...")
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    tr_idx = tr_inds.detach().cpu().numpy()
    val_idx = val_inds.detach().cpu().numpy()
    te_idx = te_inds.detach().cpu().numpy()

    # Output directory
    splits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    npz_path = os.path.join(splits_dir, f"{args.data}_split.npz")
    np.savez_compressed(npz_path, tr_inds=tr_idx, val_inds=val_idx, te_inds=te_idx)
    print(f"Saved split indices to: {npz_path}")

    # Calculate day cutoffs (i, j) identical to data_loading.py
    timestamps = te_data.timestamps
    n_days = int(timestamps.max().item() / (3600 * 24) + 1)
    daily_trans = []
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        cnt = int(((timestamps >= l) & (timestamps < r)).sum().item())
        daily_trans.append(cnt)

    d_ts = np.array(daily_trans)
    split_per = [0.6, 0.2, 0.2]
    split_scores = dict()
    for di, dj in itertools.combinations(range(len(d_ts)), 2):
        split_totals = [d_ts[:di].sum(), d_ts[di:dj].sum(), d_ts[dj:].sum()]
        split_totals_sum = np.sum(split_totals)
        split_props = [v / split_totals_sum for v in split_totals]
        split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
        split_scores[(di, dj)] = max(split_error)

    i, j = min(split_scores, key=split_scores.get)

    all_y = te_data.y.detach().cpu().numpy()
    all_ts = te_data.timestamps.detach().cpu().numpy()

    def get_stats(indices, name, day_range):
        n = len(indices)
        n_laundering = int(all_y[indices].sum())
        pct_laundering = (n_laundering / n * 100.0) if n > 0 else 0.0
        ts_min = float(all_ts[indices].min()) if n > 0 else 0.0
        ts_max = float(all_ts[indices].max()) if n > 0 else 0.0
        return {
            "split": name,
            "days": day_range,
            "num_rows": n,
            "pct_total_rows": round(n / len(all_y) * 100.0, 2),
            "num_laundering": n_laundering,
            "pct_laundering": round(pct_laundering, 4),
            "timestamp_min": round(ts_min, 1),
            "timestamp_max": round(ts_max, 1),
            "duration_hours": round((ts_max - ts_min) / 3600.0, 2)
        }

    train_stats = get_stats(tr_idx, "Train", f"0 to {i-1}")
    val_stats = get_stats(val_idx, "Validation", f"{i} to {j-1}")
    test_stats = get_stats(te_idx, "Test", f"{j} to {n_days-1}")

    summary_data = {
        "dataset": args.data,
        "total_transactions": len(all_y),
        "total_laundering": int(all_y.sum()),
        "total_pct_laundering": round(float(all_y.sum()) / len(all_y) * 100.0, 4),
        "total_days": n_days,
        "day_cutoffs": {
            "i": int(i),
            "j": int(j),
            "train_days": f"0..{i-1}",
            "val_days": f"{i}..{j-1}",
            "test_days": f"{j}..{n_days-1}"
        },
        "splits": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats
        }
    }

    json_path = os.path.join(splits_dir, f"{args.data}_split_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved split summary to: {json_path}")

    # Print Table
    table_rows = [
        [
            s["split"],
            s["days"],
            f"{s['num_rows']:,}",
            f"{s['pct_total_rows']}%",
            f"{s['num_laundering']:,}",
            f"{s['pct_laundering']:.3f}%",
            f"{s['timestamp_min']:.0f}s - {s['timestamp_max']:.0f}s",
            f"{s['duration_hours']:.1f}h"
        ]
        for s in [train_stats, val_stats, test_stats]
    ]

    headers = ["Split", "Days", "Rows", "% of Total", "Laundering Rows", "Laundering %", "Timestamp Range", "Duration"]
    print("\n" + "="*80)
    print(f"SPLIT SUMMARY: {args.data} (Day cutoffs: i={i}, j={j})")
    print("="*80)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))
    print("="*80 + "\n")

if __name__ == "__main__":
    export_splits()

"""Graph feature engineering for IBM AML transaction data.

Provides leakage-safe dynamic features computed along the full chronological
transaction stream and split-isolated static features (PageRank and degrees)
computed strictly on the training partition of each split.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------

DYNAMIC_FEATURE_COLS: list[str] = [
    # Sender features (prior-only)
    "s_out_cnt",
    "s_out_amt",
    "s_in_cnt",
    "s_in_amt",
    "s_uniq_receivers",
    "s_uniq_senders_in",
    "s_gap_out",
    "s_amt_vs_mean",
    "s_net_flow",
    # Receiver features (prior-only)
    "r_in_cnt",
    "r_in_amt",
    "r_out_cnt",
    "r_out_amt",
    "r_uniq_senders",
    "r_uniq_receivers_out",
    "r_gap_in",
    # Pair features (prior-only)
    "pair_prior_cnt",
    "reverse_prior_cnt",
    # History flags
    "has_sender_history",
    "has_receiver_history",
]

STATIC_FEATURE_COLS: list[str] = [
    "pr_sender",
    "pr_receiver",
    "s_weighted_out_degree",
    "s_weighted_in_degree",
    "r_weighted_out_degree",
    "r_weighted_in_degree",
]


# ---------------------------------------------------------------------------
# Vectorized Dynamic Aggregation Helpers
# ---------------------------------------------------------------------------

def _build_event_index(
    event_key: np.ndarray, event_pos: np.ndarray, P: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorts events by composite key (event_key * P + event_pos).

    Args:
        event_key: 1D int64 array of event keys (e.g. account or pair ID).
        event_pos: 1D int64 array of event positions (0 to N-1).
        P: Composite multiplier strictly greater than max(event_pos).

    Returns:
        Tuple of (order, sorted_comp, sorted_key).
    """
    comp = event_key * P + event_pos
    order = np.argsort(comp)
    sorted_comp = comp[order]
    sorted_key = event_key[order]
    return order, sorted_comp, sorted_key


def _prior_agg(
    order: np.ndarray,
    sorted_comp: np.ndarray,
    event_val: np.ndarray,
    query_key: np.ndarray,
    query_pos: np.ndarray,
    P: int,
) -> np.ndarray:
    """Computes sum of event_val over earlier events with matching key.

    For each query row i, sums event_val over all events with
    event_key == query_key[i] and event_pos < query_pos[i].
    """
    n = len(order)
    sorted_val = event_val[order]
    prefix = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(sorted_val, out=prefix[1:])

    idx_start = np.searchsorted(sorted_comp, query_key * P, side="left")
    idx_end = np.searchsorted(sorted_comp, query_key * P + query_pos, side="left")
    return prefix[idx_end] - prefix[idx_start]


def _prior_gap(
    order: np.ndarray,
    sorted_comp: np.ndarray,
    sorted_key: np.ndarray,
    event_time: np.ndarray,
    query_key: np.ndarray,
    query_pos: np.ndarray,
    query_time: np.ndarray,
    P: int,
) -> np.ndarray:
    """Computes time difference (seconds) since the previous event with matching key.

    Returns 0.0 if no prior event exists for query_key.
    """
    sorted_time = event_time[order]
    idx_end = np.searchsorted(sorted_comp, query_key * P + query_pos, side="left")
    idx_prev = idx_end - 1
    valid_prev = idx_prev >= 0
    clamped_prev = np.maximum(idx_prev, 0)
    has_prev = valid_prev & (sorted_key[clamped_prev] == query_key)
    prev_time = np.where(has_prev, sorted_time[clamped_prev], query_time)
    gap = np.where(has_prev, query_time - prev_time, 0.0)
    return gap


# ---------------------------------------------------------------------------
# Dynamic Feature Builder
# ---------------------------------------------------------------------------

def build_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes leakage-safe dynamic features along the chronological stream.

    For row i, uses strictly rows with position < i. Same-second ties are
    resolved by row position. Never uses labels or row i itself.

    Args:
        df: DataFrame containing at least: 'from_id', 'to_id', 'Timestamp',
            'Amount Sent', 'Amount Received'.

    Returns:
        DataFrame containing exactly DYNAMIC_FEATURE_COLS, indexed by df.index.
    """
    n = len(df)
    pos = np.arange(n, dtype=np.int64)
    from_id = df["from_id"].to_numpy().astype(np.int64)
    to_id = df["to_id"].to_numpy().astype(np.int64)
    amt_sent = df["Amount Sent"].to_numpy().astype(np.float64)
    amt_recv = df["Amount Received"].to_numpy().astype(np.float64)
    ts = df["Timestamp"].to_numpy().astype(np.int64)

    P = int(max(10_000_000, n + 1))
    max_acc = max(from_id.max(), to_id.max()) if n > 0 else 0
    shift = 21 if max_acc < (1 << 20) else int(np.ceil(np.log2(max_acc + 1))) + 1
    pair_code = (from_id << shift) | to_id
    rev_pair_code = (to_id << shift) | from_id

    # is_first_pair: 1.0 on the first chronological row of each (from_id, to_id) pair
    is_first_pair = (~pd.Series(pair_code).duplicated()).to_numpy().astype(np.float64)
    ones = np.ones(n, dtype=np.float64)

    # Pre-sort event indices once per entity type
    order_from, comp_from, key_from = _build_event_index(from_id, pos, P)
    order_to, comp_to, key_to = _build_event_index(to_id, pos, P)
    order_pair, comp_pair, key_pair = _build_event_index(pair_code, pos, P)

    # --- Sender features ---------------------------------------------------
    raw_s_out_cnt = _prior_agg(order_from, comp_from, ones, from_id, pos, P)
    raw_s_out_amt = _prior_agg(order_from, comp_from, amt_sent, from_id, pos, P)
    raw_s_in_cnt = _prior_agg(order_to, comp_to, ones, from_id, pos, P)
    raw_s_in_amt = _prior_agg(order_to, comp_to, amt_recv, from_id, pos, P)
    raw_s_uniq_receivers = _prior_agg(order_from, comp_from, is_first_pair, from_id, pos, P)
    raw_s_uniq_senders_in = _prior_agg(order_to, comp_to, is_first_pair, from_id, pos, P)
    raw_s_gap_out = _prior_gap(order_from, comp_from, key_from, ts, from_id, pos, ts, P)

    mean_prior_out = np.zeros(n, dtype=np.float64)
    np.divide(raw_s_out_amt, raw_s_out_cnt, out=mean_prior_out, where=raw_s_out_cnt > 0)
    s_amt_vs_mean = np.zeros(n, dtype=np.float64)
    np.divide(amt_sent, mean_prior_out, out=s_amt_vs_mean, where=mean_prior_out > 0)

    net_diff = raw_s_in_amt - raw_s_out_amt
    s_net_flow = np.sign(net_diff) * np.log1p(np.abs(net_diff))

    # --- Receiver features -------------------------------------------------
    raw_r_in_cnt = _prior_agg(order_to, comp_to, ones, to_id, pos, P)
    raw_r_in_amt = _prior_agg(order_to, comp_to, amt_recv, to_id, pos, P)
    raw_r_out_cnt = _prior_agg(order_from, comp_from, ones, to_id, pos, P)
    raw_r_out_amt = _prior_agg(order_from, comp_from, amt_sent, to_id, pos, P)
    raw_r_uniq_senders = _prior_agg(order_to, comp_to, is_first_pair, to_id, pos, P)
    raw_r_uniq_receivers_out = _prior_agg(order_from, comp_from, is_first_pair, to_id, pos, P)
    raw_r_gap_in = _prior_gap(order_to, comp_to, key_to, ts, to_id, pos, ts, P)

    # --- Pair features -----------------------------------------------------
    raw_pair_prior_cnt = _prior_agg(order_pair, comp_pair, ones, pair_code, pos, P)
    raw_rev_prior_cnt = _prior_agg(order_pair, comp_pair, ones, rev_pair_code, pos, P)

    # --- History flags -----------------------------------------------------
    has_sender_history = ((raw_s_out_cnt + raw_s_in_cnt) > 0).astype(np.float64)
    has_receiver_history = ((raw_r_in_cnt + raw_r_out_cnt) > 0).astype(np.float64)

    # Construct output DataFrame with log1p on heavy-tailed quantities
    out = pd.DataFrame(index=df.index)
    out["s_out_cnt"] = np.log1p(raw_s_out_cnt)
    out["s_out_amt"] = np.log1p(raw_s_out_amt)
    out["s_in_cnt"] = np.log1p(raw_s_in_cnt)
    out["s_in_amt"] = np.log1p(raw_s_in_amt)
    out["s_uniq_receivers"] = np.log1p(raw_s_uniq_receivers)
    out["s_uniq_senders_in"] = np.log1p(raw_s_uniq_senders_in)
    out["s_gap_out"] = np.log1p(raw_s_gap_out)
    out["s_amt_vs_mean"] = s_amt_vs_mean
    out["s_net_flow"] = s_net_flow

    out["r_in_cnt"] = np.log1p(raw_r_in_cnt)
    out["r_in_amt"] = np.log1p(raw_r_in_amt)
    out["r_out_cnt"] = np.log1p(raw_r_out_cnt)
    out["r_out_amt"] = np.log1p(raw_r_out_amt)
    out["r_uniq_senders"] = np.log1p(raw_r_uniq_senders)
    out["r_uniq_receivers_out"] = np.log1p(raw_r_uniq_receivers_out)
    out["r_gap_in"] = np.log1p(raw_r_gap_in)

    out["pair_prior_cnt"] = np.log1p(raw_pair_prior_cnt)
    out["reverse_prior_cnt"] = np.log1p(raw_rev_prior_cnt)

    out["has_sender_history"] = has_sender_history
    out["has_receiver_history"] = has_receiver_history

    return out[DYNAMIC_FEATURE_COLS]


# ---------------------------------------------------------------------------
# PageRank & Static Feature Builder
# ---------------------------------------------------------------------------

def compute_pagerank(
    edges_from: np.ndarray,
    edges_to: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes weighted PageRank on a directed graph via scipy.sparse power iteration.

    Edge weight is the frequency of directed transactions between each node pair.

    Args:
        edges_from: 1D array of source account IDs.
        edges_to: 1D array of target account IDs.
        damping: Random teleport probability damping factor (default 0.85).
        max_iter: Maximum power iterations (default 50).
        tol: L1 convergence tolerance (default 1e-8).

    Returns:
        Tuple of (unique_nodes, pagerank_probs, weighted_out_degree, weighted_in_degree).
    """
    nodes, inv = np.unique(np.concatenate([edges_from, edges_to]), return_inverse=True)
    N = len(nodes)
    n_edges = len(edges_from)
    from_idx = inv[:n_edges]
    to_idx = inv[n_edges:]

    weights = np.ones(n_edges, dtype=np.float64)
    A = sp.csr_matrix((weights, (from_idx, to_idx)), shape=(N, N))

    out_degree = np.array(A.sum(axis=1)).flatten()
    in_degree = np.array(A.sum(axis=0)).flatten()
    non_zero = out_degree > 0

    inv_deg = np.zeros(N, dtype=np.float64)
    inv_deg[non_zero] = 1.0 / out_degree[non_zero]
    D_inv = sp.diags(inv_deg)
    M = D_inv.dot(A)
    MT = M.T.tocsr()

    p = np.full(N, 1.0 / N, dtype=np.float64)
    for _ in range(max_iter):
        p_prev = p.copy()
        s = p[non_zero].sum()
        dangling_teleport = (1.0 - damping * s) / N
        p = damping * MT.dot(p_prev) + dangling_teleport
        err = np.abs(p - p_prev).sum()
        if err < tol:
            break

    return nodes, p, out_degree, in_degree


def build_static_features(df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Computes static graph features for ALL rows using only training edges.

    Constructs a weighted directed graph from the transactions at train_idx,
    runs weighted PageRank, and assigns PageRank and in/out degrees to the
    sender and receiver of every transaction in df. Accounts absent from the
    train graph receive 0.

    Args:
        df: Full DataFrame of transactions.
        train_idx: Array of integer row indices belonging to the train split.

    Returns:
        DataFrame containing exactly STATIC_FEATURE_COLS, indexed by df.index.
    """
    df_tr = df.iloc[train_idx]
    edges_from = df_tr["from_id"].to_numpy().astype(np.int64)
    edges_to = df_tr["to_id"].to_numpy().astype(np.int64)

    nodes, p, out_deg, in_deg = compute_pagerank(edges_from, edges_to)
    N_train = len(nodes)

    all_from = df["from_id"].to_numpy()
    all_to = df["to_id"].to_numpy()

    # Fast direct array lookup if account IDs are contiguous integers
    max_node = max(all_from.max(), all_to.max()) if len(df) > 0 else 0
    if max_node < 5_000_000 and np.issubdtype(all_from.dtype, np.integer):
        pr_lookup = np.zeros(max_node + 1, dtype=np.float64)
        out_deg_lookup = np.zeros(max_node + 1, dtype=np.float64)
        in_deg_lookup = np.zeros(max_node + 1, dtype=np.float64)

        pr_lookup[nodes] = np.log10(p * N_train)
        out_deg_lookup[nodes] = np.log1p(out_deg)
        in_deg_lookup[nodes] = np.log1p(in_deg)

        pr_sender = pr_lookup[all_from]
        pr_receiver = pr_lookup[all_to]
        s_out_degree = out_deg_lookup[all_from]
        s_in_degree = in_deg_lookup[all_from]
        r_out_degree = out_deg_lookup[all_to]
        r_in_degree = in_deg_lookup[all_to]
    else:
        # Fallback for arbitrary or non-contiguous IDs
        pr_map = dict(zip(nodes, np.log10(p * N_train)))
        out_map = dict(zip(nodes, np.log1p(out_deg)))
        in_map = dict(zip(nodes, np.log1p(in_deg)))

        pr_sender = np.array([pr_map.get(x, 0.0) for x in all_from], dtype=np.float64)
        pr_receiver = np.array([pr_map.get(x, 0.0) for x in all_to], dtype=np.float64)
        s_out_degree = np.array([out_map.get(x, 0.0) for x in all_from], dtype=np.float64)
        s_in_degree = np.array([in_map.get(x, 0.0) for x in all_from], dtype=np.float64)
        r_out_degree = np.array([out_map.get(x, 0.0) for x in all_to], dtype=np.float64)
        r_in_degree = np.array([in_map.get(x, 0.0) for x in all_to], dtype=np.float64)

    out = pd.DataFrame(index=df.index)
    out["pr_sender"] = pr_sender
    out["pr_receiver"] = pr_receiver
    out["s_weighted_out_degree"] = s_out_degree
    out["s_weighted_in_degree"] = s_in_degree
    out["r_weighted_out_degree"] = r_out_degree
    out["r_weighted_in_degree"] = r_in_degree

    return out[STATIC_FEATURE_COLS]


# ---------------------------------------------------------------------------
# CLI Generation Routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Generates and saves dynamic and static graph feature parquets."""
    base_dir = Path(__file__).parent
    sys.path.insert(0, str(base_dir))
    from data_access import get_data_dir, load_formatted

    data_dir = Path(get_data_dir()) / "Small_HI"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading formatted transactions...")
    t0 = time.time()
    df = load_formatted("Small_HI")
    print(f"Loaded {len(df):,} transactions in {time.time() - t0:.2f}s")

    # Verify timestamp monotonicity
    assert df["Timestamp"].is_monotonic_increasing, "Timestamp must be non-decreasing!"

    # 1. Build and save dynamic features
    dyn_path = data_dir / "graph_dynamic.parquet"
    print("\n--- Building Dynamic Graph Features ---")
    t_dyn_start = time.time()
    df_dyn = build_dynamic_features(df)
    t_dyn_build = time.time() - t_dyn_start

    t_save_start = time.time()
    df_dyn.to_parquet(dyn_path, index=False)
    t_save = time.time() - t_save_start

    dyn_mem_mb = df_dyn.memory_usage(deep=True).sum() / (1024 * 1024)
    file_size_mb = os.path.getsize(dyn_path) / (1024 * 1024)
    print(f"Dynamic features built in {t_dyn_build:.2f}s (saved to {dyn_path} in {t_save:.2f}s)")
    print(f"Memory: {dyn_mem_mb:.1f} MB in RAM, {file_size_mb:.1f} MB on disk")

    # History coverage analysis on temporal split
    splits_dir = base_dir / "splits"
    temp_npz = splits_dir / "Small_HI_split.npz"
    if temp_npz.is_file():
        sp_data = np.load(temp_npz)
        tr_inds = sp_data["tr_inds"]
        val_inds = sp_data["val_inds"]
        te_inds = sp_data["te_inds"]

        print("\n--- Temporal Split History Coverage ---")
        for split_name, inds in [("Train", tr_inds), ("Val", val_inds), ("Test", te_inds)]:
            sub = df_dyn.iloc[inds]
            s_no_hist = (sub["has_sender_history"] == 0).mean() * 100.0
            r_no_hist = (sub["has_receiver_history"] == 0).mean() * 100.0
            print(f"  {split_name:<5}: Sender no-history: {s_no_hist:5.2f}% | Receiver no-history: {r_no_hist:5.2f}%")

    # 2. Build and save static features for temporal and random splits
    for kind in ["temporal", "random"]:
        npz_name = f"Small_HI_split.npz" if kind == "temporal" else f"Small_HI_random_split.npz"
        npz_path = splits_dir / npz_name
        if not npz_path.is_file():
            print(f"Skipping static {kind} (archive not found: {npz_path})")
            continue

        sp_data = np.load(npz_path)
        train_idx = sp_data["tr_inds"]

        static_out_path = data_dir / f"graph_static_{kind}.parquet"
        print(f"\n--- Building Static Graph Features ({kind} split) ---")
        t_stat_start = time.time()
        df_static = build_static_features(df, train_idx)
        t_stat_build = time.time() - t_stat_start
        df_static.to_parquet(static_out_path, index=False)
        print(f"Static {kind} features built in {t_stat_build:.2f}s (saved to {static_out_path})")


if __name__ == "__main__":
    main()

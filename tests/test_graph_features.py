"""Unit tests for leakage-safe graph feature engineering.

Verifies:
1. NAIVE CHECK: dynamic features match a slow reference loop across all 20 columns.
2. FUTURE-INVARIANCE: modifying future rows has zero impact on earlier rows (positions 0..1000).
3. NO-SELF/LABEL: output contains no label column, and row 0 features and flags are all strictly 0.
4. PAGERANK: sparse PageRank implementation matches networkx.pagerank within 1e-6.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_features import (
    DYNAMIC_FEATURE_COLS,
    STATIC_FEATURE_COLS,
    build_dynamic_features,
    build_static_features,
    compute_pagerank,
)


def _slow_reference_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """Slow iterative reference implementation for dynamic features."""
    records = []
    n = len(df)
    for i in range(n):
        s = df.iloc[i]["from_id"]
        r = df.iloc[i]["to_id"]
        amt_s = df.iloc[i]["Amount Sent"]
        t = df.iloc[i]["Timestamp"]
        prior = df.iloc[:i]

        p_s_out = prior[prior["from_id"] == s]
        p_s_in = prior[prior["to_id"] == s]
        p_r_in = prior[prior["to_id"] == r]
        p_r_out = prior[prior["from_id"] == r]
        p_pair = prior[(prior["from_id"] == s) & (prior["to_id"] == r)]
        p_rev = prior[(prior["from_id"] == r) & (prior["to_id"] == s)]

        raw_s_out_cnt = len(p_s_out)
        raw_s_out_amt = p_s_out["Amount Sent"].sum()
        raw_s_in_cnt = len(p_s_in)
        raw_s_in_amt = p_s_in["Amount Received"].sum()
        raw_s_uniq_recv = p_s_out["to_id"].nunique()
        raw_s_uniq_send_in = p_s_in["from_id"].nunique()
        raw_s_gap_out = (t - p_s_out.iloc[-1]["Timestamp"]) if len(p_s_out) > 0 else 0.0
        mean_out = (raw_s_out_amt / raw_s_out_cnt) if raw_s_out_cnt > 0 else 0.0
        s_amt_vs_mean = (amt_s / mean_out) if mean_out > 0 else 0.0
        diff = raw_s_in_amt - raw_s_out_amt
        s_net_flow = np.sign(diff) * np.log1p(abs(diff))

        raw_r_in_cnt = len(p_r_in)
        raw_r_in_amt = p_r_in["Amount Received"].sum()
        raw_r_out_cnt = len(p_r_out)
        raw_r_out_amt = p_r_out["Amount Sent"].sum()
        raw_r_uniq_send = p_r_in["from_id"].nunique()
        raw_r_uniq_recv_out = p_r_out["to_id"].nunique()
        raw_r_gap_in = (t - p_r_in.iloc[-1]["Timestamp"]) if len(p_r_in) > 0 else 0.0

        raw_pair_cnt = len(p_pair)
        raw_rev_cnt = len(p_rev)

        has_s_hist = 1.0 if (raw_s_out_cnt + raw_s_in_cnt) > 0 else 0.0
        has_r_hist = 1.0 if (raw_r_in_cnt + raw_r_out_cnt) > 0 else 0.0

        records.append({
            "s_out_cnt": np.log1p(raw_s_out_cnt),
            "s_out_amt": np.log1p(raw_s_out_amt),
            "s_in_cnt": np.log1p(raw_s_in_cnt),
            "s_in_amt": np.log1p(raw_s_in_amt),
            "s_uniq_receivers": np.log1p(raw_s_uniq_recv),
            "s_uniq_senders_in": np.log1p(raw_s_uniq_send_in),
            "s_gap_out": np.log1p(raw_s_gap_out),
            "s_amt_vs_mean": s_amt_vs_mean,
            "s_net_flow": s_net_flow,
            "r_in_cnt": np.log1p(raw_r_in_cnt),
            "r_in_amt": np.log1p(raw_r_in_amt),
            "r_out_cnt": np.log1p(raw_r_out_cnt),
            "r_out_amt": np.log1p(raw_r_out_amt),
            "r_uniq_senders": np.log1p(raw_r_uniq_send),
            "r_uniq_receivers_out": np.log1p(raw_r_uniq_recv_out),
            "r_gap_in": np.log1p(raw_r_gap_in),
            "pair_prior_cnt": np.log1p(raw_pair_cnt),
            "reverse_prior_cnt": np.log1p(raw_rev_cnt),
            "has_sender_history": has_s_hist,
            "has_receiver_history": has_r_hist,
        })
    return pd.DataFrame(records)


def _generate_synthetic_data(n_rows: int = 2000, n_accounts: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    from_ids = rng.randint(0, n_accounts, n_rows)
    to_ids = rng.randint(0, n_accounts, n_rows)
    # Avoid self-loops
    same = from_ids == to_ids
    to_ids[same] = (to_ids[same] + 1) % n_accounts

    # Strictly increasing timestamps
    ts_steps = rng.randint(1, 30, n_rows)
    ts = np.cumsum(ts_steps) + 1000

    amt_sent = np.round(rng.exponential(250.0, n_rows) + 5.0, 2)
    amt_recv = np.round(amt_sent * rng.uniform(0.98, 1.02, n_rows), 2)
    labels = (rng.rand(n_rows) < 0.05).astype(int)

    return pd.DataFrame({
        "from_id": from_ids,
        "to_id": to_ids,
        "Timestamp": ts,
        "Amount Sent": amt_sent,
        "Amount Received": amt_recv,
        "Is Laundering": labels,
    })


def test_naive_check():
    """a) NAIVE CHECK: 2000 synthetic rows equal slow reference loop across all features."""
    df = _generate_synthetic_data(n_rows=2000, n_accounts=50, seed=42)
    fast_df = build_dynamic_features(df)
    slow_df = _slow_reference_dynamic(df)

    assert list(fast_df.columns) == DYNAMIC_FEATURE_COLS
    for col in DYNAMIC_FEATURE_COLS:
        max_diff = np.max(np.abs(fast_df[col].values - slow_df[col].values))
        assert max_diff < 1e-7, f"Column '{col}' differs from reference: max_diff={max_diff:.2e}"
    print("test_naive_check PASSED: all 20 dynamic features match slow reference loop.")


def test_future_invariance():
    """b) FUTURE-INVARIANCE: modifying rows > 1000 leaves rows 0..1000 unchanged."""
    df_orig = _generate_synthetic_data(n_rows=2000, n_accounts=50, seed=100)
    feat_orig = build_dynamic_features(df_orig)

    df_mod = df_orig.copy()
    # Scramble amounts, accounts, and labels after position 1000
    n = len(df_mod)
    rng = np.random.RandomState(999)
    df_mod.loc[1001:, "from_id"] = rng.randint(100, 200, n - 1001)
    df_mod.loc[1001:, "to_id"] = rng.randint(200, 300, n - 1001)
    df_mod.loc[1001:, "Amount Sent"] = rng.exponential(1000.0, n - 1001)
    df_mod.loc[1001:, "Amount Received"] = rng.exponential(1000.0, n - 1001)
    df_mod.loc[1001:, "Is Laundering"] = 1

    feat_mod = build_dynamic_features(df_mod)

    # Check rows 0..1000 are invariant to future modifications
    sub_orig = feat_orig.iloc[:1001].to_numpy()
    sub_mod = feat_mod.iloc[:1001].to_numpy()
    max_diff = np.max(np.abs(sub_orig - sub_mod))
    assert np.allclose(sub_orig, sub_mod, atol=1e-8, rtol=1e-8), (
        f"Future leak detected! Rows 0..1000 changed after modifying rows > 1000 (max_diff={max_diff:.2e})"
    )
    print(f"test_future_invariance PASSED: zero leakage from future into past rows (max_diff={max_diff:.2e} < 1e-8).")


def test_no_self_label():
    """c) NO-SELF/LABEL: no label column, and row 0 has all-zero history features and flags."""
    df = _generate_synthetic_data(n_rows=500, n_accounts=20, seed=77)
    feat = build_dynamic_features(df)

    # Output must not contain label
    for col in feat.columns:
        col_lower = col.lower()
        assert "laundering" not in col_lower and "label" not in col_lower, f"Label leaked into column '{col}'"

    # Row 0 must be all 0.0
    row_0 = feat.iloc[0].to_dict()
    for col, val in row_0.items():
        assert val == 0.0, f"Row 0 has non-zero history feature '{col}': {val}"

    assert row_0["has_sender_history"] == 0.0
    assert row_0["has_receiver_history"] == 0.0
    print("test_no_self_label PASSED: no labels present and row 0 is completely zeroed.")


def test_pagerank():
    """d) PAGERANK: sparse PageRank matches networkx.pagerank within 1e-6."""
    rng = np.random.RandomState(42)
    n_edges = 1500
    u = rng.randint(0, 150, n_edges)
    v = rng.randint(0, 150, n_edges)

    # Build NetworkX weighted graph
    G = nx.DiGraph()
    for src, dst in zip(u, v):
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1.0
        else:
            G.add_edge(src, dst, weight=1.0)

    nodes, p, out_deg, in_deg = compute_pagerank(u, v, damping=0.85, max_iter=200, tol=1e-12)
    nx_pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-12, weight="weight")
    nx_p = np.array([nx_pr[node] for node in nodes])

    max_diff = np.max(np.abs(p - nx_p))
    assert max_diff < 1e-6, f"PageRank max diff {max_diff:.2e} exceeds tolerance 1e-6"
    print(f"test_pagerank PASSED: PageRank max diff with NetworkX is {max_diff:.2e} (< 1e-6).")


if __name__ == "__main__":
    test_naive_check()
    test_future_invariance()
    test_no_self_label()
    test_pagerank()
    print("\nALL GRAPH FEATURE TESTS PASSED SUCCESSFULLY!")

"""Per-row tabular feature engineering for IBM AML transaction data.

All features are derived strictly from a single transaction row with no
aggregation over other rows and no future information. Safe for both
temporal and random splits.

Columns used from formatted_transactions.csv:
    Amount Sent, Amount Received  -- float, the raw transaction amounts
    Sent Currency, Received Currency, Payment Format  -- int category codes
    Timestamp  -- int, seconds since dataset epoch (2022-09-01 00:00:00)

Columns explicitly excluded:
    EdgeID  -- database surrogate key, not a transaction attribute
    from_id, to_id  -- node IDs belonging to the graph pipeline only
    Timestamp (raw)  -- used only for deriving cyclical features, not as input
    Is Laundering  -- the label; never an input feature
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public constants - reused by run_tabular.py for column ordering
# ---------------------------------------------------------------------------

# 2022-09-01 falls on a Thursday (Monday=0 convention => Thursday=3).
_EPOCH_WEEKDAY: int = 3

FEATURE_COLS: list[str] = [
    "amount_sent",
    "amount_received",
    "log1p_amount_sent",
    "log1p_amount_received",
    "amount_ratio",       # amount_sent / (amount_received + 1e-8)
    "sent_currency",
    "received_currency",
    "payment_format",
    "same_currency",
    "hour_of_day",
    "day_of_week",
]


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-row tabular features from a formatted_transactions DataFrame.

    No cross-row aggregation is performed; every feature is computable from a
    single transaction record in isolation. This design prevents temporal
    leakage in both the temporal and random split experiments.

    Args:
        df: DataFrame loaded from formatted_transactions.csv (or a row-subset
            of it). Must contain columns: 'Amount Sent', 'Amount Received',
            'Sent Currency', 'Received Currency', 'Payment Format', 'Timestamp'.

    Returns:
        New DataFrame with exactly the columns in FEATURE_COLS, matching the
        row count and index of the input. All columns are float64 so that
        downstream sklearn / LightGBM / XGBoost models receive a uniform array.
    """
    out = pd.DataFrame(index=df.index)

    # --- Raw amounts -------------------------------------------------------
    out["amount_sent"] = df["Amount Sent"].astype("float64")
    out["amount_received"] = df["Amount Received"].astype("float64")

    # --- Log-scaled amounts (log1p keeps zeros finite) --------------------
    out["log1p_amount_sent"] = np.log1p(out["amount_sent"])
    out["log1p_amount_received"] = np.log1p(out["amount_received"])

    # --- Ratio (detects FX conversion, layering, amount distortion) ------
    out["amount_ratio"] = out["amount_sent"] / (out["amount_received"] + 1e-8)

    # --- Categorical codes (already integer-encoded by format_kaggle_files.py)
    out["sent_currency"] = df["Sent Currency"].astype("float64")
    out["received_currency"] = df["Received Currency"].astype("float64")
    out["payment_format"] = df["Payment Format"].astype("float64")

    # --- Same-currency flag (1 if currencies match, else 0) ---------------
    out["same_currency"] = (
        df["Sent Currency"] == df["Received Currency"]
    ).astype("float64")

    # --- Cyclical time features derived from Timestamp --------------------
    # Timestamp is seconds elapsed since 2022-09-01 00:00:00 UTC.
    ts = df["Timestamp"].astype("int64")
    out["hour_of_day"] = ((ts % 86400) // 3600).astype("float64")
    # Day-of-week: epoch day 0 was a Thursday (index 3 in Mon=0 convention).
    out["day_of_week"] = ((ts // 86400 + _EPOCH_WEEKDAY) % 7).astype("float64")

    # Guarantee column order matches the FEATURE_COLS declaration above
    return out[FEATURE_COLS]

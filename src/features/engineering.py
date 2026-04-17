"""Feature engineering — velocity features, rolling aggregations, derived ratios."""

from __future__ import annotations

import pandas as pd


VELOCITY_WINDOWS = {
    "1h": "1h",
    "6h": "6h",
    "24h": "24h",
    "7d": "7d",
}


def add_velocity_features(
    df: pd.DataFrame,
    timestamp_col: str,
    group_col: str,
    amount_col: str = "amount",
    windows: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Count and sum transactions per group within rolling time windows."""
    df = df.copy().sort_values(timestamp_col)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    windows = windows or VELOCITY_WINDOWS

    df = df.set_index(timestamp_col)
    for label, window in windows.items():
        rolled = df.groupby(group_col)[amount_col].rolling(window, closed="left")
        df[f"txn_count_{label}"] = rolled.count().reset_index(level=0, drop=True)
        df[f"txn_sum_{label}"] = rolled.sum().reset_index(level=0, drop=True)

    return df.reset_index()


def add_ratio_features(df: pd.DataFrame, amount_col: str = "amount") -> pd.DataFrame:
    df = df.copy()
    if "median_purchase_price" in df.columns:
        df["ratio_to_median"] = df[amount_col] / (df["median_purchase_price"] + 1e-9)
    if "avg_amount_24h" in df.columns:
        df["ratio_to_24h_avg"] = df[amount_col] / (df["avg_amount_24h"] + 1e-9)
    return df


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    return df


def build_features(
    df: pd.DataFrame,
    timestamp_col: str,
    group_col: str = "card_id",
) -> pd.DataFrame:
    df = add_time_features(df, timestamp_col)
    df = add_velocity_features(df, timestamp_col, group_col)
    df = add_ratio_features(df)
    return df

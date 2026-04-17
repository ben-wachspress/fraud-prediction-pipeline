"""Preprocessing — cleaning, encoding, train/val/test split, class imbalance handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")
    df = df.drop_duplicates()
    df = df.dropna(subset=[c for c in df.columns if df[c].dtype != object])
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_columns: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    encoders = encoders or {}
    for col in categorical_columns:
        if col not in df.columns:
            continue
        if fit:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc
        else:
            enc = encoders[col]
            df[col] = enc.transform(df[col].astype(str))
    return df, encoders


def split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )
    # val carved from remaining train
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_fraction, stratify=y_train, random_state=random_seed
    )

    train = X_train.assign(**{target: y_train})
    val = X_val.assign(**{target: y_val})
    test = X_test.assign(**{target: y_test})
    return train, val, test


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.1,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_seed)
    X_res, y_res = sm.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def scale_numerics(
    df: pd.DataFrame,
    numeric_columns: list[str],
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    cols = [c for c in numeric_columns if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler

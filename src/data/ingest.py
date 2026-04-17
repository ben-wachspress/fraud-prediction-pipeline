"""Data ingestion — load raw transaction data from CSV/Parquet/DB."""

from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()


def load_transactions(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    console.log(f"Loading data from [bold]{path}[/bold]")

    loaders = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }
    loader = loaders.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df = loader(path, **kwargs)
    console.log(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

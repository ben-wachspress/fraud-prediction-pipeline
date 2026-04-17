import pandas as pd
import pytest

from src.data.preprocess import clean, encode_categoricals, split


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
        "merchant_category": ["grocery", "gas", "online", "grocery", "gas"] * 20,
        "is_fraud": [0, 0, 0, 0, 1] * 20,
    })


def test_clean_removes_duplicates(sample_df):
    df_dup = pd.concat([sample_df, sample_df.iloc[:5]])
    cleaned = clean(df_dup)
    assert len(cleaned) == len(sample_df)


def test_encode_categoricals(sample_df):
    df, encoders = encode_categoricals(sample_df, ["merchant_category"])
    assert df["merchant_category"].dtype in (int, "int64", "int32")
    assert "merchant_category" in encoders


def test_split_sizes(sample_df):
    train, val, test = split(sample_df, target="is_fraud", test_size=0.2, val_size=0.1)
    total = len(train) + len(val) + len(test)
    assert total == len(sample_df)
    assert len(test) == pytest.approx(len(sample_df) * 0.2, abs=2)

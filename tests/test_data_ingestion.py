from __future__ import annotations

import pandas as pd
import pytest

from payshap_ml.pipelines.data_ingestion import clean, validate_schema, REQUIRED_COLUMNS


def make_raw_df(**overrides) -> pd.DataFrame:
    base = {
        "transaction_id": ["t1", "t2", "t3"],
        "debtor_shap_id": ["+27821111111", "+27822222222", "+27823333333"],
        "creditor_shap_id": ["+27831111111", "+27832222222", "+27833333333"],
        "amount_zar": [100.0, 200.0, 300.0],
        "timestamp": pd.to_datetime(
            ["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z", "2026-01-01T12:00:00Z"]
        ),
        "clearing_system_property": ["pacs.008"] * 3,
        "proxy_type": ["MOBILE"] * 3,
        "is_fraud": [0, 0, 1],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_schema_passes_with_all_columns():
    df = make_raw_df()
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = make_raw_df().drop(columns=["is_fraud"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df)


def test_clean_deduplicates_transaction_ids():
    df = make_raw_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate row
    cleaned, stats = clean(df)
    assert stats["dropped_duplicates"] == 1
    assert cleaned["transaction_id"].is_unique


def test_clean_drops_null_critical_fields():
    df = make_raw_df()
    df.loc[0, "debtor_shap_id"] = None
    cleaned, stats = clean(df)
    assert stats["dropped_null_critical_fields"] == 1
    assert len(cleaned) == 2


def test_clean_drops_non_positive_amounts():
    df = make_raw_df(amount_zar=[100.0, -5.0, 0.0])
    cleaned, stats = clean(df)
    assert stats["dropped_non_positive_amount"] == 2
    assert (cleaned["amount_zar"] > 0).all()


def test_clean_sorts_by_timestamp():
    df = make_raw_df()
    df = df.iloc[::-1].reset_index(drop=True)  # reverse order
    cleaned, _ = clean(df)
    assert cleaned["timestamp"].is_monotonic_increasing

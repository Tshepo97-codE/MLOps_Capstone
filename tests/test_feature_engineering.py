from __future__ import annotations

import pandas as pd

from payshap_ml.pipelines.feature_engineering import (
    add_amount_ratio_to_history,
    add_hourly_frequency,
    add_time_delta,
    add_velocity_features,
    time_based_split,
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "debtor_shap_id": ["a", "a", "a", "b"],
            "creditor_shap_id": ["x", "y", "z", "x"],
            "amount_zar": [100.0, 110.0, 500.0, 50.0],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T10:00:00Z",
                    "2026-01-01T10:05:00Z",
                    "2026-01-01T10:10:00Z",
                    "2026-01-01T10:00:00Z",
                ]
            ),
        }
    )


def test_add_time_delta_first_transaction_gets_sentinel():
    df = add_time_delta(make_df())
    a_rows = df[df["debtor_shap_id"] == "a"].sort_values("timestamp")
    assert a_rows.iloc[0]["time_delta_seconds"] == 1e9
    assert a_rows.iloc[1]["time_delta_seconds"] == 300.0  # 5 minutes


def test_add_velocity_features_counts_only_prior_transactions():
    df = add_velocity_features(make_df(), windows_minutes=[15])
    a_rows = df[df["debtor_shap_id"] == "a"].sort_values("timestamp")
    # First transaction for 'a' has zero prior transactions in-window.
    assert a_rows.iloc[0]["velocity_15m"] == 0
    # Third transaction for 'a' has two priors within 15 minutes.
    assert a_rows.iloc[2]["velocity_15m"] == 2


def test_add_hourly_frequency_increments_per_debtor():
    df = add_hourly_frequency(make_df(), bucket_minutes=60)
    a_rows = df[df["debtor_shap_id"] == "a"].sort_values("timestamp")
    assert list(a_rows["hourly_txn_frequency"]) == [0, 1, 2]


def test_add_amount_ratio_neutral_when_insufficient_history():
    df = add_amount_ratio_to_history(make_df(), window_days=90, min_history=3)
    # No debtor has 3+ prior transactions in this tiny fixture, so every
    # row should get the neutral ratio of 1.0.
    assert (df["amount_ratio_to_hist_mean"] == 1.0).all()


def test_time_based_split_is_chronological_and_non_overlapping():
    df = make_df().sort_values("timestamp")
    train_df, test_df = time_based_split(df, "timestamp", test_size=0.25, val_size=0.1)
    assert len(train_df) + len(test_df) == len(df)
    if len(train_df) and len(test_df):
        assert train_df["timestamp"].max() <= test_df["timestamp"].min()

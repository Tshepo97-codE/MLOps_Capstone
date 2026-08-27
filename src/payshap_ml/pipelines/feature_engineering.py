"""Stage 2 — Feature engineering.

Derives PayShap / ISO 20022 pacs.008-specific features:
  * ShapID velocity — count of transfers per debtor_shap_id within
    rolling windows (5, 15, 60, 1440 minutes by default).
  * Hourly transaction frequency — bucketed count of transfers per
    debtor_shap_id per hour-of-day bucket.
  * Time-delta between transfers — seconds since the debtor's previous
    outgoing transfer.
  * Ratio-to-historical-mean-amount — current amount divided by the
    debtor's trailing mean transfer amount over a historical window.
  * Encodes select ISO 20022 message attributes (clearing system
    property, proxy type, agent BICs, purpose code, charge bearer).

All windows are computed causally (using only past transactions relative
to each row's timestamp) to avoid label leakage.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from payshap_ml.utils.config import ensure_parent_dir, get_params_arg_parser, load_params
from payshap_ml.utils.logging import get_logger

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def add_time_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds since this debtor_shap_id's previous outgoing transfer."""
    df = df.sort_values(["debtor_shap_id", "timestamp"])
    prev_ts = df.groupby("debtor_shap_id")["timestamp"].shift(1)
    df["time_delta_seconds"] = (df["timestamp"] - prev_ts).dt.total_seconds()
    # First-ever transfer for a debtor has no prior event; use a large
    # sentinel rather than NaN so tree models can split on "new payer".
    df["time_delta_seconds"] = df["time_delta_seconds"].fillna(1e9)
    return df


def add_velocity_features(df: pd.DataFrame, windows_minutes: list[int]) -> pd.DataFrame:
    """Causal rolling count of transfers per debtor_shap_id per window.

    Implemented via a per-group rolling window on the time index, counting
    prior transactions only (row itself excluded) to avoid leakage.
    """
    df = df.sort_values(["debtor_shap_id", "timestamp"]).copy()
    df = df.set_index("timestamp")

    for w in windows_minutes:
        col = f"velocity_{w}m"
        counts = (
            df.groupby("debtor_shap_id")["transaction_id"]
            .rolling(f"{w}min", closed="left")
            .count()
        )
        # groupby+rolling returns a MultiIndex (debtor_shap_id, timestamp);
        # align back to df's row order via the second index level.
        df[col] = counts.reset_index(level=0, drop=True).fillna(0).values

    df = df.reset_index()
    return df


def add_hourly_frequency(df: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    """Count of a debtor's transfers in the current hour-of-day bucket,
    computed over the debtor's full transaction history up to this row."""
    df = df.sort_values(["debtor_shap_id", "timestamp"]).copy()
    bucket = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // bucket_minutes
    df["_hour_bucket"] = bucket
    df["hourly_txn_frequency"] = (
        df.groupby(["debtor_shap_id", "_hour_bucket"]).cumcount()
    )
    df = df.drop(columns="_hour_bucket")
    return df


def add_amount_ratio_to_history(
    df: pd.DataFrame, window_days: int, min_history: int
) -> pd.DataFrame:
    """Ratio of current amount to the debtor's trailing mean amount.

    Uses an expanding-then-windowed causal mean (shifted by one row so the
    current transaction is never included in its own baseline).
    """
    df = df.sort_values(["debtor_shap_id", "timestamp"]).copy()
    df = df.set_index("timestamp")

    rolling_mean = (
        df.groupby("debtor_shap_id")["amount_zar"]
        .rolling(f"{window_days}D", closed="left")
        .mean()
    )
    rolling_count = (
        df.groupby("debtor_shap_id")["amount_zar"]
        .rolling(f"{window_days}D", closed="left")
        .count()
    )

    hist_mean = rolling_mean.reset_index(level=0, drop=True)
    hist_count = rolling_count.reset_index(level=0, drop=True)

    df = df.reset_index()
    df["_hist_mean_amount"] = hist_mean.values
    df["_hist_count"] = hist_count.values

    enough_history = df["_hist_count"] >= min_history
    df["amount_ratio_to_hist_mean"] = np.where(
        enough_history & (df["_hist_mean_amount"] > 0),
        df["amount_zar"] / df["_hist_mean_amount"],
        1.0,  # neutral ratio when there isn't enough history yet (new payer)
    )
    df = df.drop(columns=["_hist_mean_amount", "_hist_count"])
    return df


def encode_iso20022_fields(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Simple categorical -> integer-code encoding for ISO 20022 attributes
    present in the dataset. Missing configured fields are skipped with a
    warning-level log rather than raising, since not every deployment will
    populate every optional pacs.008 element.
    """
    df = df.copy()
    for field in fields:
        if field not in df.columns:
            continue
        df[field] = df[field].astype("category")
        df[f"{field}_code"] = df[field].cat.codes
    return df


def scale_numeric_features(
    df: pd.DataFrame, numeric_cols: list[str], method: str
) -> pd.DataFrame:
    if method == "none":
        return df
    scaler_cls = SCALERS.get(method)
    if scaler_cls is None:
        raise ValueError(f"Unknown scaling_method '{method}'")
    df = df.copy()
    scaler = scaler_cls()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def time_based_split(
    df: pd.DataFrame, time_col: str, test_size: float, val_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: earliest (1 - test_size) for train, remainder
    for test. Time-based splitting prevents future information leaking into
    training for a real-time fraud model. val_size is reserved for the
    training stage's internal early-stopping split, taken from the tail of
    the train partition.
    """
    df = df.sort_values(time_col)
    n = len(df)
    test_cut = int(n * (1 - test_size))
    train_df = df.iloc[:test_cut].copy()
    test_df = df.iloc[test_cut:].copy()
    return train_df, test_df


def main() -> None:
    parser = get_params_arg_parser("PayShap feature engineering stage")
    args = parser.parse_args()
    params = load_params(args.params)

    log = get_logger(__name__, params["base"]["log_level"])

    data_cfg = params["data"]
    feat_cfg = params["features"]
    split_cfg = params["split"]

    log.info("Loading interim data from %s", data_cfg["interim_path"])
    df = pd.read_parquet(data_cfg["interim_path"])

    log.info("Adding time-delta feature")
    df = add_time_delta(df)

    log.info("Adding ShapID velocity features: %s", feat_cfg["velocity"]["windows_minutes"])
    df = add_velocity_features(df, feat_cfg["velocity"]["windows_minutes"])

    log.info("Adding hourly transaction frequency")
    df = add_hourly_frequency(df, feat_cfg["hourly_frequency"]["bucket_size_minutes"])

    log.info("Adding amount-ratio-to-historical-mean feature")
    df = add_amount_ratio_to_history(
        df,
        feat_cfg["amount_ratio"]["historical_window_days"],
        feat_cfg["amount_ratio"]["min_history_transactions"],
    )

    log.info("Encoding ISO 20022 categorical fields")
    df = encode_iso20022_fields(df, feat_cfg["iso20022_fields"])

    numeric_feature_cols = [
        "amount_zar",
        "time_delta_seconds",
        "hourly_txn_frequency",
        "amount_ratio_to_hist_mean",
    ] + [f"velocity_{w}m" for w in feat_cfg["velocity"]["windows_minutes"]]

    log.info("Scaling numeric features using method=%s", feat_cfg["scaling_method"])
    df = scale_numeric_features(df, numeric_feature_cols, feat_cfg["scaling_method"])

    log.info("Splitting train/test using method=%s", split_cfg["method"])
    train_df, test_df = time_based_split(
        df,
        split_cfg["time_column"],
        data_cfg["test_size"],
        data_cfg["val_size"],
    )
    log.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    ensure_parent_dir(data_cfg["processed_train_path"])
    train_df.to_parquet(data_cfg["processed_train_path"], index=False)
    test_df.to_parquet(data_cfg["processed_test_path"], index=False)
    log.info(
        "Wrote processed train/test sets to %s and %s",
        data_cfg["processed_train_path"],
        data_cfg["processed_test_path"],
    )


if __name__ == "__main__":
    sys.exit(main())

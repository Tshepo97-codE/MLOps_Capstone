"""Stage 1 — Ingestion & cleaning.

Reads raw PayShap pacs.008-derived transaction records and produces a
cleaned, schema-validated interim dataset. This stage intentionally does
NOT engineer behavioural features (velocity, ratios, etc.) — that belongs
to feature_engineering.py so DVC can cache each stage independently.
"""

from __future__ import annotations

import sys

import pandas as pd

from payshap_ml.utils.config import ensure_parent_dir, get_params_arg_parser, load_params
from payshap_ml.utils.logging import get_logger

# Minimal schema contract for raw pacs.008-derived rows. Extend as the
# upstream ISO 20022 extraction layer exposes more fields.
REQUIRED_COLUMNS = [
    "transaction_id",
    "debtor_shap_id",
    "creditor_shap_id",
    "amount_zar",
    "timestamp",
    "clearing_system_property",
    "proxy_type",
    "is_fraud",
]


def load_raw(raw_path: str) -> pd.DataFrame:
    return pd.read_parquet(raw_path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Deduplicate on transaction_id — PayShap message replays should not
    # be double-counted as separate events.
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    dropped_dupes = before - len(df)

    # Drop rows with a null target or missing identity keys — these cannot
    # be used for supervised training or velocity feature computation.
    critical_cols = ["is_fraud", "debtor_shap_id", "creditor_shap_id", "timestamp"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped_nulls = before - len(df)

    # Normalize types.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["amount_zar"] = pd.to_numeric(df["amount_zar"], errors="coerce")
    df = df.dropna(subset=["amount_zar"])

    # Guard against non-physical amounts (PayShap enforces positive transfer
    # amounts; anything <= 0 indicates upstream extraction error, not fraud).
    before = len(df)
    df = df[df["amount_zar"] > 0]
    dropped_bad_amount = before - len(df)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df, {
        "dropped_duplicates": int(dropped_dupes),
        "dropped_null_critical_fields": int(dropped_nulls),
        "dropped_non_positive_amount": int(dropped_bad_amount),
        "rows_out": int(len(df)),
    }


def main() -> None:
    parser = get_params_arg_parser("PayShap ingestion & cleaning stage")
    args = parser.parse_args()
    params = load_params(args.params)

    log = get_logger(__name__, params["base"]["log_level"])

    data_cfg = params["data"]
    raw_path = data_cfg["raw_path"]
    interim_path = data_cfg["interim_path"]

    log.info("Loading raw data from %s", raw_path)
    df = load_raw(raw_path)
    log.info("Loaded %d raw rows", len(df))

    validate_schema(df)
    df_clean, stats = clean(df)
    log.info("Cleaning stats: %s", stats)

    ensure_parent_dir(interim_path)
    df_clean.to_parquet(interim_path, index=False)
    log.info("Wrote cleaned dataset (%d rows) to %s", len(df_clean), interim_path)


if __name__ == "__main__":
    sys.exit(main())

"""Dev-only helper: generates a synthetic raw transactions.parquet matching
the schema expected by src/payshap_ml/pipelines/data_ingestion.py.

This is NOT part of the tracked DVC pipeline — it exists so a fresh clone
of this repo can run `dvc repro` end-to-end without a real PayShap data
feed. Replace with a real ingestion source before production use.

Usage:
    python scripts/generate_synthetic_raw_data.py --rows 50000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_rows: int, fraud_rate: float, n_debtors: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    debtor_ids = [f"+2782{rng.integers(0, 10_000_000):07d}" for _ in range(n_debtors)]
    creditor_ids = [f"+2783{rng.integers(0, 10_000_000):07d}" for _ in range(n_debtors)]

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [
        start + timedelta(seconds=int(s))
        for s in np.sort(rng.integers(0, 60 * 60 * 24 * 90, size=n_rows))
    ]

    proxy_types = rng.choice(["MOBILE", "ID_NUMBER", "EMAIL", "ACCOUNT"], size=n_rows)
    bics = rng.choice(
        ["FIRNZAJJ", "SBZAZAJJ", "ABSAZAJJ", "NEDSZAJJ", "CABLZAJJ"], size=n_rows
    )
    purpose_codes = rng.choice(["CASH", "SALA", "SUPP", "UTIL"], size=n_rows)

    is_fraud = rng.random(n_rows) < fraud_rate
    # Fraudulent transactions skew toward higher amounts, roughly, to give
    # the model *some* learnable signal in this synthetic dataset.
    base_amount = rng.lognormal(mean=6.0, sigma=1.0, size=n_rows)
    amount = np.where(is_fraud, base_amount * rng.uniform(2, 6, size=n_rows), base_amount)

    df = pd.DataFrame(
        {
            "transaction_id": [f"PS-{i:010d}" for i in range(n_rows)],
            "debtor_shap_id": rng.choice(debtor_ids, size=n_rows),
            "creditor_shap_id": rng.choice(creditor_ids, size=n_rows),
            "amount_zar": np.round(amount, 2),
            "timestamp": timestamps,
            "clearing_system_property": "pacs.008",
            "proxy_type": proxy_types,
            "debtor_agent_bic": bics,
            "creditor_agent_bic": rng.choice(bics, size=n_rows),
            "purpose_code": purpose_codes,
            "charge_bearer": "SLEV",
            "is_fraud": is_fraud.astype(int),
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--fraud-rate", type=float, default=0.001)
    parser.add_argument("--n-debtors", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/transactions.parquet")
    args = parser.parse_args()

    df = generate(args.rows, args.fraud_rate, args.n_debtors, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows ({df['is_fraud'].sum()} fraud) to {out_path}")


if __name__ == "__main__":
    main()

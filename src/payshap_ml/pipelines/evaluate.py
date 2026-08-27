"""Stage 4 — Evaluation on the held-out test set.

Loads the trained model artifact and the processed test set, computes
PR-AUC, ROC-AUC, F1, precision, recall, and a batch-inference latency
sample, then writes metrics + PR/ROC curve points for DVC plots.

Note: this measures in-process model.predict latency on a sample of rows,
which is useful as a model-only signal, but is NOT the real-time serving
benchmark. For the true end-to-end HTTP latency against the FastAPI
`/predict` endpoint, use `src/payshap_ml/benchmarks/latency.py`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from payshap_ml.utils.config import ensure_parent_dir, get_params_arg_parser, load_params
from payshap_ml.utils.logging import get_logger


def measure_batch_latency(model, X: pd.DataFrame, sample_size: int) -> float:
    """Average per-row latency (ms) over `sample_size` single-row predict
    calls, approximating serial real-time inference cost."""
    sample_size = min(sample_size, len(X))
    sample = X.sample(n=sample_size, random_state=0) if sample_size < len(X) else X
    times = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def main() -> None:
    parser = get_params_arg_parser("PayShap model evaluation stage")
    args = parser.parse_args()
    params = load_params(args.params)

    log = get_logger(__name__, params["base"]["log_level"])

    data_cfg = params["data"]
    eval_cfg = params["evaluation"]

    log.info("Loading model from models/model.pkl")
    model = joblib.load("models/model.pkl")

    with open("models/feature_names.json") as f:
        feature_cols = json.load(f)

    log.info("Loading test set from %s", data_cfg["processed_test_path"])
    test_df = pd.read_parquet(data_cfg["processed_test_path"])

    target_col = data_cfg["target_column"]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    proba = model.predict_proba(X_test)[:, 1]
    threshold = eval_cfg["decision_threshold"]
    pred = (proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    log.info("Measuring per-row batch inference latency")
    latency_ms = measure_batch_latency(model, X_test, eval_cfg["latency_sample_size"])

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_single_row_latency_ms": float(latency_ms),
        "test_rows": int(len(test_df)),
        "test_positive_rate": float(y_test.mean()),
    }
    log.info("Test metrics: %s", metrics)

    eval_metrics_path = Path("reports/eval_metrics.json")
    ensure_parent_dir(eval_metrics_path)
    with eval_metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    precisions, recalls, _ = precision_recall_curve(y_test, proba)
    pd.DataFrame({"precision": precisions, "recall": recalls}).to_csv(
        "reports/pr_curve.csv", index=False
    )

    fpr, tpr, _ = roc_curve(y_test, proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv("reports/roc_curve.csv", index=False)

    log.info("Evaluation artifacts written to reports/")


if __name__ == "__main__":
    sys.exit(main())

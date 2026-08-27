"""Stage 3 — Model training with MLflow tracking and registry promotion.

Trains a gradient-boosted decision tree classifier (LightGBM or XGBoost,
selected via params.yaml) on PayShap transaction features, handling the
extreme class imbalance typical of fraud labels (~0.1% positive class)
via either cost-sensitive learning (scale_pos_weight) or SMOTE.

All hyperparameters, metrics, feature importances, and the fitted model
(with an explicit signature + input example) are logged to MLflow. Models
that clear the configured PR-AUC threshold are registered to the
MLflow Model Registry under the "Staging" stage.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from payshap_ml.utils.config import ensure_parent_dir, get_params_arg_parser, load_params
from payshap_ml.utils.logging import get_logger

NON_FEATURE_COLUMNS = {
    "transaction_id",
    "debtor_shap_id",
    "creditor_shap_id",
    "timestamp",
}


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Keep numeric/encoded columns only; drop identifiers, raw timestamp,
    the target, and raw (pre-encoding) categorical text columns whose
    `_code` counterpart is already present.
    """
    candidate_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS | {target_col}]
    feature_cols = []
    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
        # skip raw categorical/text columns — their *_code encoded version
        # (added in feature_engineering.py) is used instead.
    return feature_cols


def build_model(model_cfg: dict, algorithm: str, seed: int):
    if algorithm == "lightgbm":
        cfg = model_cfg["lightgbm"]
        return lgb.LGBMClassifier(
            objective=cfg["objective"],
            boosting_type=cfg["boosting_type"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            num_leaves=cfg["num_leaves"],
            min_child_samples=cfg["min_child_samples"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=cfg["scale_pos_weight"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            random_state=seed,
            n_jobs=-1,
        )
    elif algorithm == "xgboost":
        cfg = model_cfg["xgboost"]
        return xgb.XGBClassifier(
            objective=cfg["objective"],
            eval_metric=cfg["eval_metric"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=cfg["scale_pos_weight"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")


def apply_smote(X: pd.DataFrame, y: pd.Series, smote_cfg: dict, seed: int):
    sm = SMOTE(
        sampling_strategy=smote_cfg["sampling_strategy"],
        k_neighbors=smote_cfg["k_neighbors"],
        random_state=seed,
    )
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def flatten_params_for_mlflow(params: dict, prefix: str = "") -> dict:
    """Flatten nested params.yaml sections into dotted keys for MLflow
    log_params (which requires scalar values)."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_params_for_mlflow(v, prefix=f"{key}."))
        elif isinstance(v, list):
            flat[key] = str(v)
        else:
            flat[key] = v
    return flat


def main() -> None:
    parser = get_params_arg_parser("PayShap model training stage")
    args = parser.parse_args()
    params = load_params(args.params)

    log = get_logger(__name__, params["base"]["log_level"])

    seed = params["base"]["seed"]
    data_cfg = params["data"]
    model_cfg = params["model"]
    train_cfg = params["training"]

    np.random.seed(seed)

    mlflow.set_tracking_uri(train_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(train_cfg["mlflow"]["experiment_name"])

    log.info("Loading processed training data from %s", data_cfg["processed_train_path"])
    train_df = pd.read_parquet(data_cfg["processed_train_path"])

    target_col = data_cfg["target_column"]
    feature_cols = select_feature_columns(train_df, target_col)
    log.info("Using %d feature columns: %s", len(feature_cols), feature_cols)

    X = train_df[feature_cols]
    y = train_df[target_col].astype(int)

    # Held-out split for early stopping / validation metrics, taken from
    # the tail of the (already time-sorted) training partition.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=data_cfg["val_size"], shuffle=False
    )

    imbalance_strategy = model_cfg["imbalance_strategy"]
    if imbalance_strategy == "smote":
        log.info("Applying SMOTE oversampling to training fold")
        X_fit, y_fit = apply_smote(X_fit, y_fit, model_cfg["smote"], seed)

    algorithm = model_cfg["algorithm"]
    model = build_model(model_cfg, algorithm, seed)

    with mlflow.start_run(run_name=f"{algorithm}_train") as run:
        mlflow.log_params(flatten_params_for_mlflow(model_cfg))
        mlflow.log_param("imbalance_strategy", imbalance_strategy)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train_rows", len(X_fit))

        log.info("Fitting %s model", algorithm)
        fit_kwargs = {}
        if algorithm == "lightgbm":
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "eval_metric": "average_precision",
                "callbacks": [
                    lgb.early_stopping(model_cfg["lightgbm"]["early_stopping_rounds"]),
                    lgb.log_evaluation(period=0),
                ],
            }
        elif algorithm == "xgboost":
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
        model.fit(X_fit, y_fit, **fit_kwargs)

        # --- Validation metrics ---
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        pr_auc = average_precision_score(y_val, val_proba)
        roc_auc = roc_auc_score(y_val, val_proba)
        precision = precision_score(y_val, val_pred, zero_division=0)
        recall = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)

        # --- Single-row inference latency, measured here for a quick
        # training-time signal; the authoritative benchmark is the
        # dedicated harness in benchmarks/latency.py against the live API.
        sample = X_val.iloc[[0]]
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict_proba(sample)
        latency_ms = ((time.perf_counter() - t0) / 100) * 1000

        metrics = {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "single_row_latency_ms": float(latency_ms),
        }
        log.info("Validation metrics: %s", metrics)
        mlflow.log_metrics(metrics)

        # --- Feature importances ---
        importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
        importances_path = Path("reports/feature_importances.json")
        ensure_parent_dir(importances_path)
        with importances_path.open("w") as f:
            json.dump(importances, f, indent=2)
        mlflow.log_artifact(str(importances_path))

        # --- Signature + input example for schema enforcement at inference ---
        signature = infer_signature(X_val, val_proba)
        input_example = X_val.iloc[:5]

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # --- Persist local artifacts for the FastAPI app / DVC outs ---
        ensure_parent_dir(data_cfg.get("model_path", "models/model.pkl"))
        joblib.dump(model, "models/model.pkl")
        with open("models/feature_names.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        train_metrics_path = Path("reports/train_metrics.json")
        ensure_parent_dir(train_metrics_path)
        with train_metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        # --- Conditional registry promotion ---
        threshold = train_cfg["registry_threshold_pr_auc"]
        if pr_auc >= threshold:
            log.info(
                "PR-AUC %.4f >= threshold %.4f — registering model to Staging",
                pr_auc,
                threshold,
            )
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_name = train_cfg["mlflow"]["registered_model_name"]
            mv = mlflow.register_model(model_uri=model_uri, name=registered_name)

            client = MlflowClient()
            client.transition_model_version_stage(
                name=registered_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            log.info(
                "Registered %s version %s to stage=Staging", registered_name, mv.version
            )
        else:
            log.warning(
                "PR-AUC %.4f below threshold %.4f — model NOT registered",
                pr_auc,
                threshold,
            )


if __name__ == "__main__":
    sys.exit(main())

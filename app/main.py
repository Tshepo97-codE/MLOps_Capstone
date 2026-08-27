"""Real-time PayShap fraud-scoring API.

Loads the trained model once at startup (from the MLflow Model Registry
if configured, falling back to the local joblib artifact produced by the
training pipeline), then serves single-transaction scoring requests under
the platform's <50ms end-to-end latency budget.

Run locally with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.schemas import PayShapTransactionRequest, PayShapTransactionResponse
from payshap_ml.utils.logging import get_logger

log = get_logger("payshap_ml.app")

MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_REGISTRY_MODEL_NAME = os.getenv("MLFLOW_REGISTRY_MODEL_NAME", "PayShap_Fraud_Detector")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/model.pkl")
LOCAL_FEATURE_NAMES_PATH = os.getenv("LOCAL_FEATURE_NAMES_PATH", "models/feature_names.json")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

# Populated at startup via the lifespan handler below.
ml_state: dict[str, Any] = {
    "model": None,
    "feature_names": None,
    "model_version": "unloaded",
}


def _load_from_mlflow_registry() -> tuple[Any, list[str], str] | None:
    """Attempt to load the latest version of the registered model at the
    configured stage. Returns None (rather than raising) if MLflow is not
    reachable or no tracking URI is configured, so the app can fall back
    to a local artifact in dev / CI environments without an MLflow server.
    """
    if not MLFLOW_TRACKING_URI:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.get_latest_versions(MLFLOW_REGISTRY_MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            log.warning(
                "No model versions found for %s at stage=%s",
                MLFLOW_REGISTRY_MODEL_NAME,
                MODEL_STAGE,
            )
            return None
        latest = versions[0]
        model_uri = f"models:/{MLFLOW_REGISTRY_MODEL_NAME}/{latest.version}"
        model = mlflow.sklearn.load_model(model_uri)

        # Feature order comes from the logged signature's input schema.
        model_info = mlflow.models.get_model_info(model_uri)
        if model_info.signature is not None and model_info.signature.inputs is not None:
            feature_names = [f.name for f in model_info.signature.inputs.inputs]
        else:
            feature_names = None

        version_label = f"registry:{MLFLOW_REGISTRY_MODEL_NAME}/v{latest.version}"
        log.info("Loaded model from MLflow registry: %s", version_label)
        return model, feature_names, version_label
    except Exception as exc:  # noqa: BLE001 — deliberate broad fallback
        log.warning("Falling back to local model artifact; MLflow load failed: %s", exc)
        return None


def _load_from_local_artifact() -> tuple[Any, list[str], str]:
    model_path = Path(LOCAL_MODEL_PATH)
    feature_names_path = Path(LOCAL_FEATURE_NAMES_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"No local model artifact found at {model_path}. "
            "Run the training pipeline (`make reproduce`) first."
        )
    model = joblib.load(model_path)
    feature_names = None
    if feature_names_path.exists():
        with feature_names_path.open() as f:
            feature_names = json.load(f)
    version_label = f"local:{model_path}"
    log.info("Loaded model from local artifact: %s", version_label)
    return model, feature_names, version_label


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading fraud detection model at startup...")
    loaded = _load_from_mlflow_registry()
    if loaded is None:
        loaded = _load_from_local_artifact()

    model, feature_names, version_label = loaded
    ml_state["model"] = model
    ml_state["feature_names"] = feature_names
    ml_state["model_version"] = version_label
    log.info("Model ready: %s", version_label)

    yield

    log.info("Shutting down; clearing model state")
    ml_state["model"] = None


app = FastAPI(
    title="PayShap Real-Time Fraud Detection API",
    description="Scores instant credit transfer transactions for fraud risk.",
    version="0.1.0",
    lifespan=lifespan,
)


def build_feature_row(payload: PayShapTransactionRequest, feature_names: list[str] | None) -> pd.DataFrame:
    """Build a single-row feature frame matching the training schema.

    NOTE: this endpoint scores on request-intrinsic fields only. In
    production, ShapID velocity / historical-amount-ratio features require
    a low-latency feature store lookup (e.g. Redis) keyed on
    debtor_shap_id, which is out of scope for this scaffold but should be
    wired in here — see the TODO below — before this matches the training
    feature_engineering.py contract exactly.
    """
    row = {
        "amount_zar": payload.amount_zar,
        # TODO(feature-store): fetch real velocity / time-delta / amount-ratio
        # features for payload.debtor_shap_id from the online feature store.
        # Placeholder neutral defaults are used here so the endpoint is
        # runnable end-to-end without a live feature store dependency.
        "time_delta_seconds": 1e9,
        "hourly_txn_frequency": 0,
        "amount_ratio_to_hist_mean": 1.0,
        "velocity_5m": 0,
        "velocity_15m": 0,
        "velocity_60m": 0,
        "velocity_1440m": 0,
        "clearing_system_property_code": hash(payload.clearing_system_property) % 1000,
        "proxy_type_code": hash(payload.proxy_type.value) % 1000,
        "debtor_agent_bic_code": hash(payload.debtor_agent_bic or "") % 1000,
        "creditor_agent_bic_code": hash(payload.creditor_agent_bic or "") % 1000,
        "purpose_code_code": hash(payload.purpose_code or "") % 1000,
        "charge_bearer_code": hash(payload.charge_bearer or "") % 1000,
    }
    df = pd.DataFrame([row])
    if feature_names:
        # Reindex to the exact training-time column order; fill anything
        # the model expects that we didn't compute with 0 rather than
        # silently dropping/reordering columns.
        df = df.reindex(columns=feature_names, fill_value=0)
    return df


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if ml_state["model"] is not None else "model_not_loaded",
        "model_version": ml_state["model_version"],
    }


@app.post("/predict", response_model=PayShapTransactionResponse)
async def predict(payload: PayShapTransactionRequest, response: Response):
    if ml_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    features = build_feature_row(payload, ml_state["feature_names"])
    proba = float(ml_state["model"].predict_proba(features)[:, 1][0])
    is_high_risk = proba >= DECISION_THRESHOLD

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Inference-Time-MS"] = f"{elapsed_ms:.3f}"

    return PayShapTransactionResponse(
        transaction_id=payload.transaction_id,
        fraud_score=proba,
        is_high_risk=is_high_risk,
        decision_threshold=DECISION_THRESHOLD,
        model_version=ml_state["model_version"],
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal scoring error"})

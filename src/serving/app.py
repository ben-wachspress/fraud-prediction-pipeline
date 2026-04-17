"""FastAPI inference service for fraud prediction."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

app = FastAPI(title="Fraud Prediction API", version="1.0")

MODEL_PATH = Path("models/best_model.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.json")


@lru_cache(maxsize=1)
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
    return model, feature_names


class TransactionRequest(BaseModel):
    transaction_id: str
    features: dict[str, float]

    @field_validator("features")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("features must not be empty")
        return v


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float = 0.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    model, feature_names = _load_artifacts()

    missing = set(feature_names) - set(request.features)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    X = np.array([[request.features[f] for f in feature_names]])
    fraud_proba = float(model.predict_proba(X)[0, 1])
    threshold = 0.5

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(fraud_proba, 6),
        is_fraud=fraud_proba >= threshold,
        threshold=threshold,
    )

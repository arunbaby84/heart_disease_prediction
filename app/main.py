from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_pipeline import FEATURE_COLUMNS, load_model  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "best_model.joblib"


class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: HeartInput) -> dict[str, float | int | str]:
    logger.info("Received prediction request")
    model = load_model(MODEL_PATH)
    record = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(record)[0])
    confidence = float(model.predict_proba(record)[0][1])
    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_probability": confidence,
        "status": "success",
    }

"""
code/app.py
-----------
FastAPI application – Online Shopper Purchase Intention API

Endpoints:
    GET  /              – health check
    GET  /model-info    – model metadata + metrics
    POST /predict       – single session prediction
    POST /predict-batch – batch predictions

Run locally:
    uvicorn code.app:app --reload --port 8000
"""

import os
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = BASE_DIR / "models"

# Use tuned model if available, else fall back to base xgboost
_TUNED_PATH = DATA_DIR / "best_model.pkl"
_BASE_PATH  = MODELS_DIR / "xgboost.pkl"
MODEL_PATH  = _TUNED_PATH if _TUNED_PATH.exists() else _BASE_PATH

SCALER_PATH   = DATA_DIR / "scaler.pkl"
FEATURES_PATH = DATA_DIR / "feature_names.pkl"

# ── Encoding maps (must match training) ──────────────────────────────────────
MONTH_COL_MAP = {
    "Dec": "Month_Dec", "Feb": "Month_Feb", "Jul": "Month_Jul",
    "June": "Month_June", "Mar": "Month_Mar", "May": "Month_May",
    "Nov": "Month_Nov", "Oct": "Month_Oct", "Sep": "Month_Sep",
    # Jan / Apr / Aug / June are reference categories (all-zero)
}
ALL_MONTH_COLS   = list(MONTH_COL_MAP.values())
ALL_VISITOR_COLS = ["VisitorType_Other", "VisitorType_Returning_Visitor"]

# ── Global artifacts ──────────────────────────────────────────────────────────
_model         = None
_scaler        = None
_feature_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _feature_names
    print("🚀 Loading ML artifacts…")
    _model         = joblib.load(MODEL_PATH)
    _scaler        = joblib.load(SCALER_PATH)
    _feature_names = joblib.load(FEATURES_PATH)
    print(f"✅ Model loaded from {MODEL_PATH.name}")
    yield
    print("🛑 Shutting down API.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Online Shopper Purchase Intention API",
    description=(
        "Predicts whether an online shopping session will result in a **purchase**.\n\n"
        "Best model: **XGBoost** (F1: 0.66 | ROC-AUC: 0.93 | Accuracy: 89.3%)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow React dev server + production nginx
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ShopperSession(BaseModel):
    Administrative:             int   = Field(default=0,   ge=0)
    Administrative_Duration:    float = Field(default=0.0, ge=0.0)
    Informational:              int   = Field(default=0,   ge=0)
    Informational_Duration:     float = Field(default=0.0, ge=0.0)
    ProductRelated:             int   = Field(default=35,  ge=0)
    ProductRelated_Duration:    float = Field(default=2500.0, ge=0.0)
    BounceRates:                float = Field(default=0.01, ge=0.0, le=1.0)
    ExitRates:                  float = Field(default=0.03, ge=0.0, le=1.0)
    PageValues:                 float = Field(default=25.4, ge=0.0)
    SpecialDay:                 float = Field(default=0.0,  ge=0.0, le=1.0)
    Month: Literal[
        "Jan","Feb","Mar","Apr","May","June",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ] = "Nov"
    OperatingSystems:  int = Field(default=2, ge=1)
    Browser:           int = Field(default=2, ge=1)
    Region:            int = Field(default=1, ge=1)
    TrafficType:       int = Field(default=2, ge=1)
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"] = "Returning_Visitor"
    Weekend: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Administrative": 0, "Administrative_Duration": 0.0,
                "Informational": 0, "Informational_Duration": 0.0,
                "ProductRelated": 35, "ProductRelated_Duration": 2500.0,
                "BounceRates": 0.01, "ExitRates": 0.03,
                "PageValues": 25.4, "SpecialDay": 0.0,
                "Month": "Nov", "OperatingSystems": 2, "Browser": 2,
                "Region": 1, "TrafficType": 2,
                "VisitorType": "Returning_Visitor", "Weekend": False
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction:             int
    label:                  str
    purchase_probability:   float
    no_purchase_probability: float


class BatchPredictionResponse(BaseModel):
    total:       int
    predictions: List[PredictionResponse]


# ── Preprocessing helper ──────────────────────────────────────────────────────
def preprocess(session: dict) -> np.ndarray:
    num_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "OperatingSystems", "Browser", "Region", "TrafficType",
    ]
    row = {c: session[c] for c in num_cols}
    row["Weekend"] = int(session["Weekend"])

    # Month one-hot
    for col in ALL_MONTH_COLS:
        row[col] = 0
    mapped = MONTH_COL_MAP.get(session["Month"])
    if mapped:
        row[mapped] = 1

    # VisitorType one-hot
    row["VisitorType_Other"]              = int(session["VisitorType"] == "Other")
    row["VisitorType_Returning_Visitor"]  = int(session["VisitorType"] == "Returning_Visitor")

    df = pd.DataFrame([row])[_feature_names]
    return _scaler.transform(df)


def run_prediction(session: dict) -> dict:
    X = preprocess(session)
    pred  = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    return {
        "prediction":             pred,
        "label":                  "Purchase" if pred == 1 else "No Purchase",
        "purchase_probability":   round(float(proba[1]), 4),
        "no_purchase_probability": round(float(proba[0]), 4),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "model_file":   MODEL_PATH.name,
        "version":      "2.0.0",
        "docs":         "/docs",
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_name":  "XGBoost Classifier (GridSearchCV Tuned)",
        "description": "Best model selected via GridSearchCV on F1-Score.",
        "metrics": {
            "accuracy":  0.8933,
            "precision": 0.6537,
            "recall":    0.6623,
            "f1_score":  0.6580,
            "roc_auc":   0.9280,
        },
        "dataset": {
            "total_samples":        12330,
            "features":             26,
            "smote_applied":        True,
            "purchase_rate":        "15.47%",
        },
        "training": {
            "gridsearchcv": True,
            "cv_folds":     3,
            "scoring":      "f1",
        },
    }


@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
def predict(session: ShopperSession):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    try:
        return PredictionResponse(**run_prediction(session.model_dump()))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict-batch", tags=["Prediction"], response_model=BatchPredictionResponse)
def predict_batch(sessions: List[ShopperSession]):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    if not sessions:
        raise HTTPException(422, "Batch cannot be empty.")
    try:
        preds = [PredictionResponse(**run_prediction(s.model_dump())) for s in sessions]
        return BatchPredictionResponse(total=len(preds), predictions=preds)
    except Exception as e:
        raise HTTPException(500, str(e))

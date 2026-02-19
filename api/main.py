"""
Online Shopper Purchase Intention – FastAPI Application
-------------------------------------------------------
Run the API:
    uvicorn api.main:app --reload

Then open:
    http://localhost:8000/docs   → Swagger UI  (interactive testing)
    http://localhost:8000/redoc  → ReDoc       (clean documentation)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import List

from api.schemas import (
    ShopperSession,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
)
from api.model import load_artifacts, preprocess_session, make_prediction


# ─── App-level state: loaded once at startup ────────────────────────────────────
_model = None
_scaler = None
_feature_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts when the API starts; clean up on shutdown."""
    global _model, _scaler, _feature_names
    print("🚀 Loading ML artifacts…")
    _model, _scaler, _feature_names = load_artifacts()
    print("✅ XGBoost model, scaler and feature names loaded successfully.")
    yield
    print("🛑 API shutting down.")


# ─── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Online Shopper Purchase Intention API",
    description=(
        "## 🛒 Purchase Intent Prediction\n\n"
        "Predict whether an online shopping session will result in a **purchase** "
        "using a trained **XGBoost** model (Best Model Choice).\n\n"
        "### Available Endpoints\n"
        "| Method | Path | Description |\n"
        "|--------|------|-------------|\n"
        "| `GET`  | `/` | Health check |\n"
        "| `GET`  | `/model-info` | Model details & metrics |\n"
        "| `POST` | `/predict` | Single session prediction |\n"
        "| `POST` | `/predict-batch` | Batch predictions |\n\n"
        "---\n"
        "*Course: Python for Data Science – Guided Machine Learning | Week 3*"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 – Health Check
# ─────────────────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    tags=["Health"],
    summary="Health check",
    response_description="API status and version",
)
def root():
    """
    **Health check** – confirms the API is running and the model is loaded.
    """
    return {
        "status": "ok",
        "message": "Online Shopper Purchase Intention API is running ✅",
        "version": "1.0.0",
        "model_loaded": _model is not None,
        "docs": "http://localhost:8000/docs",
    }


# ─────────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 – Model Info
# ─────────────────────────────────────────────────────────────────────────────────

@app.get(
    "/model-info",
    tags=["Model"],
    summary="Get deployed model information",
    response_model=ModelInfoResponse,
)
def model_info():
    """
    Returns information about the deployed model:
    - Model name and description
    - Week 2 evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    - Dataset statistics
    """
    return ModelInfoResponse(
        model_name="XGBoost Classifier",
        description=(
            "Gradient Boosting model trained on the Online Shoppers Purchasing "
            "Intention dataset. Selected as best model in Week 2 based on "
            "F1-Score and ROC-AUC."
        ),
        week2_metrics={
            "accuracy": 0.8933,
            "precision": 0.6537,
            "recall": 0.6623,
            "f1_score": 0.6580,
            "roc_auc": 0.9280,
        },
        dataset={
            "total_samples": 12330,
            "features": 26,
            "train_size": 9864,
            "test_size": 2466,
            "smote_applied": True,
            "balanced_train_size": 16676,
            "purchase_rate_original": "15.47%",
        },
        how_to_use=(
            "POST a ShopperSession JSON to /predict to get a purchase prediction. "
            "Use /predict-batch to send a list of sessions at once."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 – Single Prediction
# ─────────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict purchase intent for one session",
    response_model=PredictionResponse,
)
def predict(session: ShopperSession):
    """
    Accepts a **single** shopping session and returns:
    - `prediction`: `1` (Purchase) or `0` (No Purchase)
    - `label`: human-readable label
    - `purchase_probability`: confidence score for purchase (0–1)
    - `no_purchase_probability`: confidence score for no purchase (0–1)

    **Example input** is pre-filled in the Swagger UI – just click *Try it out*!
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    try:
        X = preprocess_session(session.model_dump(), _scaler, _feature_names)
        result = make_prediction(X, _model)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 – Batch Prediction
# ─────────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict-batch",
    tags=["Prediction"],
    summary="Predict purchase intent for multiple sessions",
    response_model=BatchPredictionResponse,
)
def predict_batch(sessions: List[ShopperSession]):
    """
    Accepts a **list** of shopping sessions and returns a prediction for each.

    Useful for:
    - Evaluating multiple customers at once
    - Batch scoring from a database
    - Demonstrating the API's scalability

    **Maximum recommended batch size:** 1000 sessions.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    if len(sessions) == 0:
        raise HTTPException(status_code=422, detail="Batch cannot be empty.")

    try:
        predictions = []
        for session in sessions:
            X = preprocess_session(session.model_dump(), _scaler, _feature_names)
            result = make_prediction(X, _model)
            predictions.append(PredictionResponse(**result))

        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

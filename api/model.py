"""
Model loader and preprocessing helper.
Loads the trained XGBoost model, scaler, and feature names once at startup.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths (relative to project root) ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"
SCALER_PATH = BASE_DIR / "data" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "data" / "feature_names.pkl"

# Month columns created by pd.get_dummies(drop_first=True) on the training data
# Reference month dropped: "Aug" (alphabetically first)
MONTH_COLUMNS = [
    "Month_Dec", "Month_Feb", "Month_Jul", "Month_June",
    "Month_Mar", "Month_May", "Month_Nov", "Month_Oct", "Month_Sep"
]

# VisitorType columns created by pd.get_dummies(drop_first=True)
# Reference category dropped: "New_Visitor"
VISITOR_COLUMNS = [
    "VisitorType_Other", "VisitorType_Returning_Visitor"
]


def load_artifacts():
    """Load all model artifacts. Called once at API startup."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names


def preprocess_session(session_data: dict, scaler, feature_names: list) -> np.ndarray:
    """
    Convert raw ShopperSession dict into a scaled feature array
    ready for the XGBoost model.

    Replicates exactly the same steps done in code/preprocessing.py:
      1. One-Hot encode Month and VisitorType
      2. Binary encode Weekend
      3. Scale all features with the fitted scaler
    """

    # ── 1. Start with numerical features ────────────────────────────────────────
    numerical_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "OperatingSystems", "Browser", "Region", "TrafficType"
    ]
    row = {col: session_data[col] for col in numerical_cols}

    # ── 2. Weekend: bool → int ───────────────────────────────────────────────────
    row["Weekend"] = int(session_data["Weekend"])

    # ── 3. One-Hot encode Month ──────────────────────────────────────────────────
    month = session_data["Month"]
    # Map user-facing month names to dataset column names
    month_col_map = {
        "Dec": "Month_Dec", "Feb": "Month_Feb", "Jul": "Month_Jul",
        "June": "Month_June", "Mar": "Month_Mar", "May": "Month_May",
        "Nov": "Month_Nov", "Oct": "Month_Oct", "Sep": "Month_Sep"
    }
    for col in MONTH_COLUMNS:
        row[col] = 0
    if month in month_col_map:
        row[month_col_map[month]] = 1
    # Aug is the reference (dropped) category → all Month cols = 0

    # ── 4. One-Hot encode VisitorType ────────────────────────────────────────────
    visitor = session_data["VisitorType"]
    # Map directly using known column names
    row["VisitorType_Other"] = 1 if visitor == "Other" else 0
    row["VisitorType_Returning_Visitor"] = 1 if visitor == "Returning_Visitor" else 0
    # New_Visitor is the reference (dropped) → both cols = 0

    # ── 5. Build a DataFrame aligned to training feature names ───────────────────
    df = pd.DataFrame([row])[feature_names]

    # ── 6. Scale ─────────────────────────────────────────────────────────────────
    X_scaled = scaler.transform(df)

    return X_scaled


def make_prediction(X_scaled: np.ndarray, model) -> dict:
    """Run inference and return prediction + probabilities."""
    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]
    purchase_prob = float(probabilities[1])
    no_purchase_prob = float(probabilities[0])

    return {
        "prediction": prediction,
        "label": "Purchase" if prediction == 1 else "No Purchase",
        "purchase_probability": round(purchase_prob, 4),
        "no_purchase_probability": round(no_purchase_prob, 4),
    }

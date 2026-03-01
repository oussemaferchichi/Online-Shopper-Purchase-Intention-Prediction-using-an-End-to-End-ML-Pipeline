"""
code/modeling.py
----------------
Advanced Model Training with GridSearchCV Hyperparameter Tuning
Tracks all experiments with MLFlow.

Run:
    python -m code.modeling
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
PLOTS_DIR  = "data/plots"
EXPERIMENT = "Online Shopper – GridSearchCV Tuning"
CV_FOLDS   = 3          # keep fast; increase to 5 for production
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))
    print(f"Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),        4),
        "precision": round(precision_score(y_test, y_pred),       4),
        "recall":    round(recall_score(y_test, y_pred),          4),
        "f1_score":  round(f1_score(y_test, y_pred),              4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),        4),
    }, confusion_matrix(y_test, y_pred)


def save_confusion_matrix(cm, name):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Purchase", "Purchase"],
                yticklabels=["No Purchase", "Purchase"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    path = os.path.join(PLOTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    return path


# ── GridSearchCV + MLFlow ─────────────────────────────────────────────────────
def tune_and_log(name, estimator, param_grid, X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}\nGridSearchCV — {name}\n{'='*60}")
    mlflow.set_experiment(EXPERIMENT)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    with mlflow.start_run(run_name=f"{name} – GridSearch"):
        t0 = time.time()
        grid.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        best_model  = grid.best_estimator_
        best_params = grid.best_params_
        cv_best_f1  = round(grid.best_score_, 4)

        # Evaluate on hold-out test set
        metrics, cm = evaluate(best_model, X_test, y_test)
        cm_path     = save_confusion_matrix(cm, name)

        # Log to MLFlow
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", name)
        mlflow.log_param("cv_folds",   CV_FOLDS)
        mlflow.log_metric("cv_best_f1",    cv_best_f1)
        mlflow.log_metric("training_time", elapsed)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"✅ Best params : {best_params}")
        print(f"   CV F1       : {cv_best_f1}")
        print(f"   Test F1     : {metrics['f1_score']}")
        print(f"   ROC-AUC     : {metrics['roc_auc']}")

    return best_model, metrics, best_params


# ── Param grids ───────────────────────────────────────────────────────────────
XGB_GRID = {
    "n_estimators":  [100, 200],
    "max_depth":     [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],
}

RF_GRID = {
    "n_estimators":    [100, 200],
    "max_depth":       [10, 20, None],
    "min_samples_split": [2, 5],
}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # --- XGBoost ---
    xgb_base = XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
    )
    xgb_best, xgb_metrics, xgb_params = tune_and_log(
        "XGBoost", xgb_base, XGB_GRID, X_train, X_test, y_train, y_test
    )
    joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_tuned.pkl"))
    print("✅ Saved xgboost_tuned.pkl")

    # --- Random Forest ---
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_best, rf_metrics, rf_params = tune_and_log(
        "Random Forest", rf_base, RF_GRID, X_train, X_test, y_train, y_test
    )
    joblib.dump(rf_best, os.path.join(MODELS_DIR, "random_forest_tuned.pkl"))
    print("✅ Saved random_forest_tuned.pkl")

    # --- Pick overall best model ---
    if xgb_metrics["f1_score"] >= rf_metrics["f1_score"]:
        best_name, best_model = "XGBoost (tuned)", xgb_best
    else:
        best_name, best_model = "Random Forest (tuned)", rf_best

    joblib.dump(best_model, os.path.join(DATA_DIR, "best_model.pkl"))
    print(f"\n🏆 Best model overall: {best_name}")
    print("✅ Saved to data/best_model.pkl")

    # --- Comparison ---
    print("\n" + "="*60)
    print("COMPARISON TABLE (tuned models)")
    print("="*60)
    df = pd.DataFrame([
        {"Model": "XGBoost (tuned)",       **xgb_metrics},
        {"Model": "Random Forest (tuned)", **rf_metrics},
    ])
    print(df.to_string(index=False))
    df.to_csv(os.path.join(MODELS_DIR, "tuned_model_comparison.csv"), index=False)

"""
code/modeling.py
----------------
Advanced Model Tuning with GridSearchCV + MLFlow

Uses HistGradientBoostingClassifier (equivalent to XGBoost, zero Windows issues).

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

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data"
MODELS_DIR   = "models"
PLOTS_DIR    = "data/plots"
EXPERIMENT   = "Online Shopper – GridSearchCV Tuning"
CV_FOLDS     = 3
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


def load_data():
    X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))
    print(f"Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "f1_score":  round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
    }, confusion_matrix(y_test, y_pred)


def save_cm(cm, name):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Purchase", "Purchase"],
                yticklabels=["No Purchase", "Purchase"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    path = os.path.join(PLOTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    return path


def tune_and_log(name, estimator, param_grid, X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}\nGridSearchCV — {name}\n{'='*60}")
    mlflow.set_experiment(EXPERIMENT)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(estimator, param_grid, scoring="f1", cv=cv, n_jobs=1, verbose=1, refit=True)

    with mlflow.start_run(run_name=f"{name} – GridSearch"):
        t0 = time.time()
        grid.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)
        best_model  = grid.best_estimator_
        best_params = grid.best_params_
        cv_f1       = round(grid.best_score_, 4)
        metrics, cm = evaluate(best_model, X_test, y_test)
        cm_path     = save_cm(cm, name)
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", name)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_metric("cv_best_f1",    cv_f1)
        mlflow.log_metric("training_time", elapsed)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(best_model, "model")
        print(f"✅ Best params: {best_params}")
        print(f"   CV F1: {cv_f1}  |  Test F1: {metrics['f1_score']}  |  ROC-AUC: {metrics['roc_auc']}")
    return best_model, metrics, best_params


# ── Param grids ───────────────────────────────────────────────────────────────
HGB_GRID = {
    "max_iter":     [100, 200],
    "max_depth":    [4, 6, 8],
    "learning_rate":[0.05, 0.1],
    "l2_regularization": [0, 0.1],
}

RF_GRID = {
    "n_estimators":      [100, 200],
    "max_depth":         [10, 20, None],
    "min_samples_split": [2, 5],
}


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # HistGradientBoosting
    hgb_best, hgb_m, _ = tune_and_log(
        "HistGradientBoosting",
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        HGB_GRID, X_train, X_test, y_train, y_test
    )
    joblib.dump(hgb_best, os.path.join(MODELS_DIR, "histgb_tuned.pkl"))
    print("✅ Saved histgb_tuned.pkl")

    # Random Forest
    rf_best, rf_m, _ = tune_and_log(
        "Random Forest",
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        RF_GRID, X_train, X_test, y_train, y_test
    )
    joblib.dump(rf_best, os.path.join(MODELS_DIR, "random_forest_tuned.pkl"))
    print("✅ Saved random_forest_tuned.pkl")

    # Pick best
    if hgb_m["f1_score"] >= rf_m["f1_score"]:
        best_name, best_model = "HistGradientBoosting (tuned)", hgb_best
    else:
        best_name, best_model = "Random Forest (tuned)", rf_best

    joblib.dump(best_model, os.path.join(DATA_DIR, "best_model.pkl"))
    print(f"\n🏆 Best model: {best_name}")
    print("✅ Saved → data/best_model.pkl")

    df = pd.DataFrame([
        {"Model": "HistGradientBoosting (tuned)", **hgb_m},
        {"Model": "Random Forest (tuned)",        **rf_m},
    ])
    print("\n" + df.to_string(index=False))
    df.to_csv(os.path.join(MODELS_DIR, "tuned_model_comparison.csv"), index=False)

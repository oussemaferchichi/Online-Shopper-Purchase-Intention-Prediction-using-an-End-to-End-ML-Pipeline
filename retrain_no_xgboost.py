"""
retrain_no_xgboost.py
---------------------
Trains ALL 4 models (no XGBoost) and logs them to MLflow for comparison.
Marks the best model in the MLflow Model Registry.

Run:
    python retrain_no_xgboost.py
Then view results:
    mlflow ui  →  http://localhost:5000
"""

import os, time, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.linear_model  import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import (
    RandomForestClassifier, HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
PLOTS_DIR  = "data/plots"
EXPERIMENT = "Online Shopper – All Models Comparison"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = joblib.load(f"{DATA_DIR}/X_train.pkl")
X_test  = joblib.load(f"{DATA_DIR}/X_test.pkl")
y_train = joblib.load(f"{DATA_DIR}/y_train.pkl")
y_test  = joblib.load(f"{DATA_DIR}/y_test.pkl")
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(model):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "f1_score":  round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
    }, confusion_matrix(y_test, y_pred)

def save_cm_plot(cm, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    path = f"{PLOTS_DIR}/cm_{name.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path

def train_and_log(name, model, params, category):
    print(f"\n{'─'*50}\nTraining: {name}")
    with mlflow.start_run(run_name=name) as run:
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        metrics, cm = evaluate(model)
        cm_path = save_cm_plot(cm, name)

        # Tags for easy filtering in MLflow UI
        mlflow.set_tag("model_category", category)
        mlflow.set_tag("model_name",     name)
        mlflow.set_tag("uses_smote",     "true")
        mlflow.set_tag("is_best",        "false")   # updated later

        # Params & metrics
        mlflow.log_params({**params, "training_time_s": elapsed,
                           "train_samples": len(y_train),
                           "test_samples":  len(y_test)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=f"shopper_{name.replace(' ', '_').lower()}")

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  Time     : {elapsed}s")

    return run.info.run_id, metrics, model

# ── Define models ─────────────────────────────────────────────────────────────
MODELS = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
        {"max_iter": 1000, "solver": "lbfgs"},
        "Baseline"
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42),
        {"max_depth": 10, "min_samples_split": 10},
        "Baseline"
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1),
        {"n_estimators": 100, "max_depth": 15},
        "Ensemble"
    ),
    (
        "HistGradientBoosting",
        HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42),
        {"max_iter": 200, "max_depth": 6, "learning_rate": 0.1},
        "Ensemble"
    ),
]

# ── Train all ─────────────────────────────────────────────────────────────────
results = []
for name, model, params, category in MODELS:
    run_id, metrics, trained_model = train_and_log(name, model, params, category)
    results.append({
        "name": name, "run_id": run_id,
        "metrics": metrics, "model": trained_model
    })
    joblib.dump(trained_model, f"{MODELS_DIR}/{name.lower().replace(' ', '_')}.pkl")

# ── Pick best by F1 ───────────────────────────────────────────────────────────
best = max(results, key=lambda r: r["metrics"]["f1_score"])
print(f"\n{'='*50}")
print(f"🏆 Best model: {best['name']}  (F1={best['metrics']['f1_score']:.4f})")

# Mark best run in MLflow UI
client.set_tag(best["run_id"], "is_best", "true")
client.set_tag(best["run_id"], "model_category", f"Ensemble ⭐ BEST")

# Save best model for the API
joblib.dump(best["model"], f"{DATA_DIR}/best_model.pkl")
print(f"✅ Saved best model → data/best_model.pkl")

# ── Comparison table ──────────────────────────────────────────────────────────
print(f"\n{'='*50}\nMODEL COMPARISON TABLE\n{'='*50}")
df = pd.DataFrame([
    {"Model": r["name"], **{k: f"{v:.4f}" for k, v in r["metrics"].items()}}
    for r in results
])
print(df.to_string(index=False))
df.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)
print(f"\n✅ Comparison saved → models/model_comparison.csv")
print("\n" + "="*50)
print("Run:  mlflow ui  →  http://localhost:5000")
print("All 4 runs are in experiment: 'Online Shopper – All Models Comparison'")
print("Filter by tag 'is_best = true' to find the best model quickly!")

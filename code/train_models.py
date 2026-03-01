"""
code/train_models.py
--------------------
Model Training Pipeline with MLFlow Experiment Tracking

Trains 4 models, logs to MLflow with comparison tags.
Best model is marked with is_best=true and saved for the API.

Run:
    python -m code.train_models
Then view:
    mlflow ui  →  http://localhost:5000
"""

import os, time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.linear_model  import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_DIR   = "data"
MODELS_DIR = "models"
PLOTS_DIR  = "data/plots"
EXPERIMENT = "Online Shopper – All Models Comparison"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ── Data loader ───────────────────────────────────────────────────────────────
def load_preprocessed_data(data_dir=DATA_DIR):
    print("Loading preprocessed data...")
    X_train = joblib.load(os.path.join(data_dir, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(data_dir, "X_test.pkl"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(data_dir, "y_test.pkl"))
    feature_names = joblib.load(os.path.join(data_dir, "feature_names.pkl"))
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n=== Evaluating {model_name} ===")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),                  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0),4),
        "recall":    round(recall_score(y_test, y_pred),                     4),
        "f1_score":  round(f1_score(y_test, y_pred),                         4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),                   4),
    }
    cm = confusion_matrix(y_test, y_pred)
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    return metrics, cm


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name, output_dir=PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ── Train + log one model ─────────────────────────────────────────────────────
def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test,
                        params, experiment_name, category="Baseline"):
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        metrics, cm = evaluate_model(model, X_test, y_test, model_name)
        cm_path = plot_confusion_matrix(cm, model_name)

        # —— Tags visible in MLflow UI ——
        mlflow.set_tag("model_category", category)
        mlflow.set_tag("model_name",     model_name)
        mlflow.set_tag("uses_smote",     "true")
        mlflow.set_tag("is_best",        "false")   # updated after all models train

        # —— Parameters & metrics ——
        mlflow.log_params({
            **params,
            "training_time_s": elapsed,
            "train_samples":   len(y_train),
            "test_samples":    len(y_test),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", elapsed)
        mlflow.log_artifact(cm_path)

        # —— Register model ——
        reg_name = f"shopper_{model_name.lower().replace(' ', '_')}"
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=reg_name)

        print(f"✅ {model_name} done in {elapsed}s")

    return run.info.run_id, model, metrics


# ── Train all models ──────────────────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test,
                     experiment_name=EXPERIMENT):
    client = MlflowClient()
    all_results = []

    configs = [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000, "solver": "lbfgs"},
            "Baseline",
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42),
            {"max_depth": 10, "min_samples_split": 10},
            "Baseline",
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, max_depth=15,
                                   random_state=42, n_jobs=1),
            {"n_estimators": 100, "max_depth": 15},
            "Ensemble",
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                           learning_rate=0.1, random_state=42),
            {"max_iter": 200, "max_depth": 6, "learning_rate": 0.1},
            "Ensemble",
        ),
    ]

    for name, model, params, category in configs:
        run_id, trained, metrics = train_and_log_model(
            model, name, X_train, X_test, y_train, y_test,
            params, experiment_name, category,
        )
        all_results.append({
            "name": name, "run_id": run_id,
            "model": trained, "metrics": metrics,
        })
        joblib.dump(trained,
                    os.path.join(MODELS_DIR,
                                 f"{name.lower().replace(' ', '_')}.pkl"))

    # Mark best run
    best = max(all_results, key=lambda r: r["metrics"]["f1_score"])
    client.set_tag(best["run_id"], "is_best",        "true")
    client.set_tag(best["run_id"], "model_category", "Ensemble ⭐ BEST")
    print(f"\n🏆 Best: {best['name']}  (F1={best['metrics']['f1_score']:.4f})")

    return (
        {r["name"]: {"model": r["model"], "metrics": r["metrics"]}
         for r in all_results},
        best,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_models(results, output_dir=MODELS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    for name, data in results.items():
        path = os.path.join(output_dir,
                            f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(data["model"], path)
        print(f"✅ Saved {name} → {path}")


def print_comparison_table(results):
    rows = [
        {"Model": n, **{k: f"{v:.4f}" for k, v in d["metrics"].items()}}
        for n, d in results.items()
    ]
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}\nMODEL COMPARISON\n{'='*60}")
    print(df.to_string(index=False))
    best = df.loc[df["f1_score"].astype(float).idxmax(), "Model"]
    print(f"\n🏆 Best model (F1): {best}")
    return df, best


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()

    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING WITH MLFLOW")
    print(f"Experiment: {EXPERIMENT}")
    print("=" * 60)

    results, best_result = train_all_models(X_train, X_test, y_train, y_test)

    # Save comparison CSV
    df, best_name = print_comparison_table(results)
    df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    print(f"\n✅ Comparison table → models/model_comparison.csv")

    # Save best model for the API
    joblib.dump(best_result["model"], os.path.join(DATA_DIR, "best_model.pkl"))
    print(f"✅ Best model ({best_result['name']}) → data/best_model.pkl")

    print(f"\n{'='*60}")
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nView in MLflow UI:")
    print("  mlflow ui  →  http://localhost:5000")
    print("\nIn the MLflow UI you can:")
    print("  • See all 4 runs in one experiment")
    print("  • Compare Accuracy / F1 / ROC-AUC side-by-side")
    print("  • Filter by tag  'is_best = true'  to find the best model")
    print("  • View confusion matrix PNG artifacts per model")
    print("  • Check registered models in the 'Models' tab")

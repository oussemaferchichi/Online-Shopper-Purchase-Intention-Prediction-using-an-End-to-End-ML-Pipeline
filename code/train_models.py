"""
Model Training Pipeline with MLFlow Experiment Tracking

This module trains multiple models and tracks experiments using MLFlow:
- Baseline models: Logistic Regression, Decision Tree
- Ensemble models: Random Forest, XGBoost
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- MLFlow tracking for model comparison
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn


def load_preprocessed_data(data_dir='data'):
    """Load preprocessed data from pickle files."""
    print("Loading preprocessed data...")
    
    X_train = joblib.load(os.path.join(data_dir, 'X_train.pkl'))
    X_test = joblib.load(os.path.join(data_dir, 'X_test.pkl'))
    y_train = joblib.load(os.path.join(data_dir, 'y_train.pkl'))
    y_test = joblib.load(os.path.join(data_dir, 'y_test.pkl'))
    feature_names = joblib.load(os.path.join(data_dir, 'feature_names.pkl'))
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance and return metrics.
    """
    print(f"\n=== Evaluating {model_name} ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metrics, cm


def plot_confusion_matrix(cm, model_name, output_dir='data/plots'):
    """Plot and save confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Purchase', 'Purchase'],
                yticklabels=['No Purchase', 'Purchase'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filepath = os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filepath


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params, experiment_name):
    """
    Train a model and log everything to MLFlow.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Set MLFlow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name):
        # Record start time
        start_time = time.time()
        
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        metrics, cm = evaluate_model(model, X_test, y_test, model_name)
        
        # Plot confusion matrix
        cm_path = plot_confusion_matrix(cm, model_name)
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("training_samples", len(y_train))
        mlflow.log_param("test_samples", len(y_test))
        
        # Log metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Log confusion matrix as artifact
        mlflow.log_artifact(cm_path)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\n✅ {model_name} training completed in {training_time:.2f}s")
        print(f"MLFlow run logged successfully")
        
    return model, metrics


def train_all_models(X_train, X_test, y_train, y_test, experiment_name="Online Shopper Purchase Intention"):
    """
    Train all models (baseline + ensemble) and track with MLFlow.
    """
    models_results = {}
    
    # 1. Logistic Regression
    print("\n" + "="*60)
    print("1. LOGISTIC REGRESSION")
    print("="*60)
    lr_params = {
        'model_type': 'Logistic Regression',
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs'
    }
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model, lr_metrics = train_and_log_model(
        lr_model, "Logistic Regression", X_train, X_test, y_train, y_test, lr_params, experiment_name
    )
    models_results['Logistic Regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # 2. Decision Tree
    print("\n" + "="*60)
    print("2. DECISION TREE")
    print("="*60)
    dt_params = {
        'model_type': 'Decision Tree',
        'max_depth': 10,
        'min_samples_split': 10,
        'random_state': 42
    }
    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
    dt_model, dt_metrics = train_and_log_model(
        dt_model, "Decision Tree", X_train, X_test, y_train, y_test, dt_params, experiment_name
    )
    models_results['Decision Tree'] = {'model': dt_model, 'metrics': dt_metrics}
    
    # 3. Random Forest
    print("\n" + "="*60)
    print("3. RANDOM FOREST")
    print("="*60)
    rf_params = {
        'model_type': 'Random Forest',
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'random_state': 42
    }
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)
    rf_model, rf_metrics = train_and_log_model(
        rf_model, "Random Forest", X_train, X_test, y_train, y_test, rf_params, experiment_name
    )
    models_results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # 4. XGBoost
    print("\n" + "="*60)
    print("4. XGBOOST")
    print("="*60)
    xgb_params = {
        'model_type': 'XGBoost',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_model, xgb_metrics = train_and_log_model(
        xgb_model, "XGBoost", X_train, X_test, y_train, y_test, xgb_params, experiment_name
    )
    models_results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    return models_results


def save_models(models_results, output_dir='models'):
    """Save all trained models to disk."""
    print(f"\n{'='*60}")
    print("Saving Models")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, data in models_results.items():
        model = data['model']
        filename = f"{model_name.replace(' ', '_').lower()}.pkl"
        filepath = os.path.join(output_dir, filename)
        joblib.dump(model, filepath)
        print(f"✅ Saved {model_name} to {filepath}")


def print_comparison_table(models_results):
    """Print comparison table of all models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*60}\n")
    
    # Create DataFrame for comparison
    comparison_data = []
    for model_name, data in models_results.items():
        metrics = data['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Find best model based on F1-Score
    best_f1_idx = df_comparison['F1-Score'].astype(float).idxmax()
    best_model_name = df_comparison.loc[best_f1_idx, 'Model']
    
    print(f"\n{'='*60}")
    print(f"🏆 Best Model (based on F1-Score): {best_model_name}")
    print(f"{'='*60}")
    
    return df_comparison, best_model_name


if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()
    
    # Train all models with MLFlow tracking
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING WITH MLFLOW")
    print("="*60)
    print(f"\nMLFlow Tracking URI: {mlflow.get_tracking_uri()}")
    print("To view experiments, run: mlflow ui")
    
    models_results = train_all_models(X_train, X_test, y_train, y_test)
    
    # Save models
    save_models(models_results)
    
    # Print comparison
    comparison_df, best_model = print_comparison_table(models_results)
    
    # Save comparison table
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print(f"\n✅ Model comparison saved to models/model_comparison.csv")
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'mlflow ui' to view experiment results")
    print("2. Navigate to http://localhost:5000 in your browser")
    print("3. Compare models and review metrics")

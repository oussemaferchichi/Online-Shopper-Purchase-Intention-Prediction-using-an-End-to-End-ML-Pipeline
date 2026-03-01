# 🛒 Online Shopper Purchase Intention Prediction
### End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange?logo=mlflow)](https://mlflow.org)
[![Docker Hub](https://img.shields.io/badge/DockerHub-oussemagd-blue?logo=docker)](https://hub.docker.com/r/oussemagd/online-shopper-api)

> **Course:** Python for Data Science – Guided Machine Learning  
> **Dataset:** [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) — 12,330 sessions, 18 features  
> **Best Model:** HistGradientBoosting (Accuracy 89.3% · F1 0.658 · ROC-AUC 0.928)

---

## 🚀 Quick Start (Full Stack with Docker)

```bash
# Clone the repository
git clone https://github.com/oussemaferchichi/Online-Shopper-Purchase-Intention-Prediction-using-an-End-to-End-ML-Pipeline.git
cd "Online Shopper Purchase Intention Prediction using an End-to-End ML Pipeline"

# Start everything (API + Frontend)
docker compose up --build

# Open in browser:
#   Frontend Dashboard  →  http://localhost:3000
#   API Swagger UI      →  http://localhost:8000/docs
```

---

## 📁 Project Structure

```
📦 Online Shopper Purchase Intention Prediction
│
├── 📂 code/                        # Python scripts
│   ├── __init__.py
│   ├── preprocessing.py            # Data cleaning, encoding, SMOTE
│   ├── train_models.py             # All 4 models + MLflow tracking
│   ├── modeling.py                 # GridSearchCV hyperparameter tuning
│   └── app.py                      # FastAPI application
│
├── 📂 frontend/                    # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx                 # Main app + navigation
│   │   ├── Dashboard.jsx           # All-models comparison + charts
│   │   ├── PredictionForm.jsx      # 17-field prediction form
│   │   ├── ResultCard.jsx          # Animated result display
│   │   ├── BatchTab.jsx            # Batch prediction interface
│   │   ├── api.js                  # API base URL config
│   │   └── index.css               # Premium dark design system
│   ├── package.json
│   └── vite.config.js
│
├── 📂 data/                        # Processed data & best model
│   ├── X_train.pkl, X_test.pkl
│   ├── y_train.pkl, y_test.pkl
│   ├── scaler.pkl, feature_names.pkl
│   └── best_model.pkl              # Best model (auto-selected)
│
├── 📂 models/                      # All trained model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── histgradientboosting.pkl
│   └── model_comparison.csv
│
├── 📂 mlruns/                      # MLflow experiment logs
├── 📂 notebooks/                   # Jupyter EDA notebook
│
├── retrain_no_xgboost.py           # Run this to train all 4 models
├── Dockerfile.api                  # Backend container
├── Dockerfile.frontend             # Frontend container (multi-stage nginx)
└── docker-compose.yml              # Full stack orchestration
```

---

## 📊 Weekly Progress

### ✅ Week 1 — Setup & EDA
- Environment setup, project structure
- EDA in `notebooks/eda.ipynb`: class distribution (85/15), feature correlations
- Identified class imbalance → solved with SMOTE in Week 2

### ✅ Week 2 — Preprocessing & Imbalance (`code/preprocessing.py`)
- One-Hot Encoding: `Month` + `VisitorType` → 26 features total
- StandardScaler for numerical features
- **SMOTE** on training set only: 9,864 → 16,676 balanced samples

### ✅ Week 3 — All Models + MLflow (`code/train_models.py`)

All 4 models tracked in **one MLflow experiment** with comparison tags:

| Model | Type | Accuracy | F1-Score | ROC-AUC |
|-------|------|----------|----------|---------|
| Logistic Regression | Baseline | 85.28% | 0.609 | 0.898 |
| Decision Tree | Baseline | 85.00% | 0.599 | 0.853 |
| Random Forest | Ensemble | 88.36% | 0.655 | 0.919 |
| **HistGradientBoosting** | **⭐ Best** | **89.33%** | **0.658** | **0.928** |

**MLflow tags per run:** `model_category`, `model_name`, `is_best`, `uses_smote`  
**Best model** automatically tagged with `is_best = true` in the UI.

```bash
# Train all 4 models + log to MLflow
python retrain_no_xgboost.py

# View comparison
mlflow ui  →  http://localhost:5000
```

### ✅ Week 3 — GridSearchCV (`code/modeling.py`)
- `HistGradientBoostingClassifier` tuned via GridSearchCV (3-fold CV, F1 scoring)
- `RandomForestClassifier` also tuned
- Best tuned model saved to `data/best_model.pkl`

### ✅ Week 4 — FastAPI Backend (`code/app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/model-info` | Best model name + metrics |
| `GET` | `/models-comparison` | All 4 models metrics (for dashboard) |
| `POST` | `/predict` | Single session prediction |
| `POST` | `/predict-batch` | Batch predictions |

```bash
python -m uvicorn code.app:app --reload --port 8000
# → http://localhost:8000/docs
```

### ✅ Week 5 — React Frontend (`frontend/`)

| Tab | Features |
|-----|----------|
| 📊 Dashboard | 5 metric cards · Grouped bar chart (all 4 models) · Comparison table · Radar chart |
| 🔮 Predict | 17-field form (3 groups) · Animated probability bars · Purchase verdict |
| ⚡ Batch | JSON textarea · Multiple sessions scored at once |

```bash
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### ✅ Week 6 — Docker Orchestration

| Service | Container | Port | Built from |
|---------|-----------|------|-----------|
| `api` | `shopper-api` | 8000 | `Dockerfile.api` |
| `frontend` | `shopper-frontend` | 3000 | `Dockerfile.frontend` |

```bash
docker compose up --build
```

---

## 🐳 Docker Guide

### Commands Reference

```bash
# ── Run ─────────────────────────────────────────────────────────
docker compose up --build          # Start + rebuild
docker compose up -d               # Start in background
docker compose down                # Stop

# ── Logs ────────────────────────────────────────────────────────
docker compose logs -f api         # Follow API logs
docker compose logs -f frontend    # Follow frontend logs

# ── Docker Hub ──────────────────────────────────────────────────
$env:PATH += ";C:\Program Files\Docker\Docker\resources\bin"
docker build -f Dockerfile.api -t oussemagd/online-shopper-api:latest .
docker push oussemagd/online-shopper-api:latest
```

### Weekly Push Workflow

```bash
# 1. Build + push Docker image
$env:PATH += ";C:\Program Files\Docker\Docker\resources\bin"
docker build -f Dockerfile.api -t oussemagd/online-shopper-api:latest -t oussemagd/online-shopper-api:week6 .
docker push oussemagd/online-shopper-api:latest
docker push oussemagd/online-shopper-api:week6

# 2. Push to GitHub
git add .
git commit -m "Weeks 2-6: All models + MLflow tags + React dashboard + Docker compose"
git push origin main
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn (LR, DT, RF, HistGradientBoosting) |
| Tuning | GridSearchCV + StratifiedKFold |
| Tracking | MLflow (tags, comparison, Model Registry) |
| Imbalance | SMOTE (imbalanced-learn) |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Frontend | React 18 + Vite + Recharts + Axios |
| Deployment | Docker + docker-compose + nginx |

---

## 📈 Evaluation Alignment

| Criterion | Weight | Implementation |
|-----------|--------|---------------|
| Data Pipeline | 20% | EDA, SMOTE (`preprocessing.py`) |
| ML Excellence | 30% | 4 models + GridSearchCV + MLflow tags/registry |
| API & UI | 30% | FastAPI 5 endpoints + React full dashboard |
| Deployment | 20% | `docker-compose.yml`, two Dockerfiles |

---

**Last Updated:** Weeks 1–6 · March 2026  
**Author:** Oussema Ferchichi | Python for Data Science – Guided ML

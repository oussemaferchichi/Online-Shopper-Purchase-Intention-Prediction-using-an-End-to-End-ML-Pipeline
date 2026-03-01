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
> **Best Model:** XGBoost (Accuracy 89.3% · F1 0.658 · ROC-AUC 0.928)

---

## 🚀 Quick Start (Full Stack with Docker)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- That's it!

### Run the Complete Stack
```bash
# 1. Clone the repository
git clone https://github.com/oussemaferchichi/Online-Shopper-Purchase-Intention-Prediction-using-an-End-to-End-ML-Pipeline.git
cd "Online Shopper Purchase Intention Prediction using an End-to-End ML Pipeline"

# 2. Start everything (API + Frontend)
docker compose up --build

# 3. Open in browser:
#    Frontend (React Dashboard) → http://localhost:3000
#    API Swagger UI             → http://localhost:8000/docs
```

### Stop the Stack
```bash
docker compose down
```

---

## 📁 Project Structure

```
📦 Online Shopper Purchase Intention Prediction
│
├── 📂 code/                        # Python scripts
│   ├── preprocessing.py            # Data cleaning, encoding, SMOTE
│   ├── train_models.py             # Baseline model training + MLflow
│   ├── modeling.py                 # GridSearchCV hyperparameter tuning
│   └── app.py                      # FastAPI application (Week 4)
│
├── 📂 frontend/                    # React + Vite app (Week 5)
│   ├── src/
│   │   ├── App.jsx                 # Main app with navigation
│   │   ├── Dashboard.jsx           # Metrics + charts dashboard
│   │   ├── PredictionForm.jsx      # 17-field prediction form
│   │   ├── ResultCard.jsx          # Animated result display
│   │   ├── BatchTab.jsx            # Batch prediction interface
│   │   ├── api.js                  # API base URL config
│   │   └── index.css               # Premium dark design system
│   ├── package.json
│   └── vite.config.js
│
├── 📂 data/                        # Serialized data & models
│   ├── online_shoppers_intention.csv
│   ├── X_train.pkl, X_test.pkl
│   ├── y_train.pkl, y_test.pkl
│   ├── scaler.pkl, feature_names.pkl
│   ├── best_model.pkl              # Best GridSearchCV model
│   └── plots/                      # Confusion matrices
│
├── 📂 models/                      # Trained model files
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── xgboost_tuned.pkl           # GridSearchCV tuned
│   ├── random_forest_tuned.pkl
│   └── model_comparison.csv
│
├── 📂 notebooks/                   # Jupyter notebooks
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── 📂 mlruns/                      # MLflow experiment logs
│
├── Dockerfile.api                  # Backend container
├── Dockerfile.frontend             # Frontend container (multi-stage)
├── docker-compose.yml              # Full stack orchestration
└── requirements.txt
```

---

## 📊 Weekly Progress

### ✅ Week 1 — Setup & EDA
- Environment configuration and project structure
- Exploratory Data Analysis (`notebooks/eda.ipynb`)
  - Class distribution: **85% No Purchase / 15% Purchase**
  - Feature correlations and seasonal patterns
  - Identified class imbalance problem → solved with SMOTE in Week 2

### ✅ Week 2 — Preprocessing & Imbalance (`code/preprocessing.py`)
- One-Hot Encoding for `Month` and `VisitorType` → 26 features
- StandardScaler for numerical features
- Stratified train/test split (80/20)
- **SMOTE** applied to training set only → 9,864 → 16,676 balanced samples

### ✅ Week 3 — Advanced Modeling & MLflow (`code/train_models.py` + `code/modeling.py`)

#### Baseline Models
| Model               | Accuracy | F1-Score | ROC-AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | 85.28%   | 0.610    | 0.898   |
| Decision Tree       | 85.00%   | 0.598    | 0.853   |

#### Ensemble Models (base)
| Model         | Accuracy | F1-Score | ROC-AUC |
|---------------|----------|----------|---------|
| Random Forest | 88.36%   | 0.655    | 0.919   |
| **XGBoost**   | **89.33%** | **0.658** | **0.928** |

#### GridSearchCV Tuned (Week 2 spec)
```bash
python -m code.modeling
```
- GridSearchCV with 3-fold cross-validation, scoring=`f1`
- Best tuned model saved to `data/best_model.pkl`
- All runs tracked in MLflow

**MLflow UI:**
```bash
mlflow ui
# → http://localhost:5000
```

### ✅ Week 4 — FastAPI Backend (`code/app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check (`status: ok, model_loaded: true`) |
| `GET`  | `/model-info` | Model name, metrics, training details |
| `POST` | `/predict` | Single session → purchase prediction |
| `POST` | `/predict-batch` | List of sessions → all predictions |

**Run locally:**
```bash
uvicorn code.app:app --reload --port 8000
# → http://localhost:8000/docs  (Swagger UI)
```

**Example prediction:**
```json
// POST /predict
{
  "ProductRelated": 35, "ProductRelated_Duration": 2500,
  "BounceRates": 0.01, "ExitRates": 0.03, "PageValues": 25.4,
  "Month": "Nov", "VisitorType": "Returning_Visitor", "Weekend": false,
  ... (other fields)
}

// Response
{
  "prediction": 1,
  "label": "Purchase",
  "purchase_probability": 0.8849,
  "no_purchase_probability": 0.1151
}
```

### ✅ Week 5 — React Frontend (`frontend/`)

Premium dark dashboard built with **React 18 + Vite**:
- **📊 Dashboard tab** — Metric cards + Recharts bar & radar charts
- **🔮 Predict tab** — 17-field form grouped by category + animated result
- **⚡ Batch tab** — JSON textarea for batch scoring

**Run locally (dev mode):**
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### ✅ Week 6 — Docker Containerization (`Dockerfile.*` + `docker-compose.yml`)

Two containers orchestrated via docker-compose:

| Service   | Container           | Port | Built from         |
|-----------|---------------------|------|--------------------|
| `api`     | `shopper-api`       | 8000 | `Dockerfile.api`   |
| `frontend`| `shopper-frontend`  | 3000 | `Dockerfile.frontend` |

---

## 🐳 Docker Guide (Complete)

### Understanding Docker for this project

```
You (browser)
    │
    ├─→ http://localhost:3000  →  [shopper-frontend container]  (nginx serving React)
    │                                         │
    └─→ http://localhost:8000  →  [shopper-api container]       (uvicorn + FastAPI + XGBoost)
```

### Commands Reference

```bash
# ── Start ───────────────────────────────────────────────────────
docker compose up              # Start (use cached images)
docker compose up --build      # Start + rebuild images first
docker compose up -d           # Start in background (detached)

# ── Stop ────────────────────────────────────────────────────────
docker compose down            # Stop and remove containers
docker compose down -v         # Stop + remove volumes

# ── Logs ────────────────────────────────────────────────────────
docker compose logs            # All service logs
docker compose logs api        # API logs only
docker compose logs -f         # Follow (live) logs

# ── Status ──────────────────────────────────────────────────────
docker compose ps              # List running containers
docker ps                      # All Docker containers

# ── Rebuild single service ───────────────────────────────────────
docker compose build api       # Rebuild API only
docker compose up --build api  # Rebuild + restart API

# ── Push to Docker Hub ──────────────────────────────────────────
$env:PATH += ";C:\Program Files\Docker\Docker\resources\bin"
docker build -f Dockerfile.api -t oussemagd/online-shopper-api:latest .
docker push oussemagd/online-shopper-api:latest
```

### Weekly Push Workflow (Code → Docker → GitHub)

```bash
# Step 1: Build and tag
docker build -f Dockerfile.api -t oussemagd/online-shopper-api:latest -t oussemagd/online-shopper-api:week6 .

# Step 2: Push to Docker Hub
$env:PATH += ";C:\Program Files\Docker\Docker\resources\bin"
docker push oussemagd/online-shopper-api:latest
docker push oussemagd/online-shopper-api:week6

# Step 3: Push to GitHub
git add .
git commit -m "Week 6: Docker compose orchestration"
git push origin main
```

### Docker Hub
Image: **[oussemagd/online-shopper-api](https://hub.docker.com/r/oussemagd/online-shopper-api)**

```bash
# Anyone can run the API with just:
docker run -p 8000:8000 oussemagd/online-shopper-api:latest
```

---

## 🛠️ Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data | Pandas, NumPy | Processing |
| Visualization | Matplotlib, Seaborn, Recharts | Charts |
| ML | Scikit-learn, XGBoost | Modeling |
| Imbalance | imbalanced-learn (SMOTE) | Class balancing |
| Tuning | GridSearchCV | Hyperparameter optimization |
| Tracking | MLflow | Experiment logging |
| API | FastAPI + Uvicorn | REST backend |
| Validation | Pydantic v2 | Input validation |
| Frontend | React 18 + Vite | Dashboard UI |
| Containerization | Docker + docker-compose | Deployment |

---

## 📈 Evaluation Alignment

| Criterion | Weight | Implementation |
|-----------|--------|---------------|
| Data Pipeline | 20% | EDA (`eda.ipynb`), SMOTE (`preprocessing.py`) |
| ML Excellence | 30% | GridSearchCV + MLflow (`modeling.py`, `train_models.py`) |
| API & UI | 30% | FastAPI (`code/app.py`) + React Dashboard (`frontend/`) |
| Deployment | 20% | `docker-compose.yml`, `Dockerfile.api`, `Dockerfile.frontend` |

---

## 📚 References

- Sakar et al. (2019). [Real-time prediction of online shoppers' purchasing intention](https://link.springer.com/article/10.1007/s00521-018-3523-0)
- [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

---

**Last Updated:** Weeks 1–6 · March 2026  
**Author:** Oussema Ferchichi | Python for Data Science – Guided ML Course

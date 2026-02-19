# Online Shoppers Purchasing Intention Prediction using an End-to-End Machine Learning Pipeline

## 📋 Project Overview

This project aims to predict whether an online shopping session will result in a purchase using machine learning techniques. By analyzing user behavior data from online shopping sessions, we build a predictive model that helps e-commerce businesses understand and anticipate customer purchasing decisions.

> **Course**: Python for Data Science – Guided Machine Learning  
> **Week**: 1 - Exploratory Data Analysis & Project Setup  
> **Status**: ✅ Initial EDA Complete

---

## 🎯 Objectives

1. **Predict purchase intent**: Build a binary classification model to predict whether a shopping session will end in a purchase (Revenue = TRUE/FALSE)
2. **Analyze customer behavior**: Understand patterns and features that indicate higher purchase likelihood
3. **Handle class imbalance**: Address the significant imbalance in the dataset (~15% purchase rate) using appropriate techniques
4. **Deploy an end-to-end pipeline**: Create a complete ML workflow from data preprocessing to model deployment

---

## 📊 Dataset Description

- **Source**: [UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- **Format**: CSV
- **Target Variable**: `Revenue` (TRUE = purchase, FALSE = no purchase)
- **Key Features**:
  - **Administrative, Informational, Product-related pages**: Number of pages visited
  - **Duration metrics**: Time spent on different page types
  - **Bounce Rates & Exit Rates**: User engagement metrics
  - **Page Values**: Average value of pages visited
  - **Special Day**: Closeness to special occasions
  - **Month**: Month of the year
  - **Operating System, Browser, Region, Traffic Type**: Technical and geographic features
  - **Visitor Type**: New, Returning, or Other
  - **Weekend**: Whether the session occurred on a weekend

### ⚠️ Class Imbalance
The dataset exhibits significant class imbalance with approximately **15% purchase rate** and **85% non-purchase rate**. This imbalance is a critical challenge that will be addressed using **SMOTE (Synthetic Minority Over-sampling Technique)** during model training.

---

## 📁 Project Structure

```
Online Shopper Purchase Intention Prediction using an End-to-End ML Pipeline/
│
├── code/                           # Source code for preprocessing and modeling
│   ├── preprocessing.py            # Data preprocessing pipeline
│   └── train_models.py             # Model training with MLFlow tracking
│
├── data/                           # Dataset and visualizations
│   ├── online_shoppers_intention.csv
│   ├── plots/                      # EDA and model visualizations
│   ├── X_train.pkl                 # Preprocessed training features
│   ├── X_test.pkl                  # Preprocessed test features
│   ├── y_train.pkl                 # Training labels
│   ├── y_test.pkl                  # Test labels
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── feature_names.pkl           # Feature names after encoding
│
├── models/                         # Trained models and results
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── model_comparison.csv        # Performance comparison table
│
├── mlruns/                         # MLFlow experiment tracking data
│
├── notebooks/                      # Jupyter notebooks
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   └── preprocessing_and_modeling.ipynb  # Interactive modeling (optional)
│
├── frontend/                       # Future: React-based web interface
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Week 1-2 Progress

### ✅ Week 1: Completed Tasks

1. **Project Initialization**
   - Created organized directory structure
   - Set up data and visualization folders

2. **Exploratory Data Analysis** (`notebooks/eda.ipynb`)
   - Loaded and inspected dataset (shape, columns, data types)
   - Analyzed missing values
   - Visualized class distribution (Revenue)
   - Analyzed purchase patterns by Visitor Type and Month
   - Explored distributions of key numerical features
   - Created correlation heatmap focusing on Revenue
   - **All plots saved to `data/plots/`**

3. **Preprocessing Plan Documentation**
   - Documented comprehensive preprocessing strategy
   - Defined pipeline approach using scikit-learn

### 📝 Key Findings from EDA

- **Class Imbalance Confirmed**: ~15% purchase rate requires SMOTE
- **Visitor Type Impact**: Different purchase behaviors across visitor segments
- **Seasonal Patterns**: Purchase rates vary by month
- **Feature Correlations**: PageValues shows strong correlation with Revenue

---

### ✅ Week 2: Completed Tasks

1. **Data Preprocessing Pipeline** (`code/preprocessing.py`)
   - ✅ No missing values detected in dataset
   - ✅ Categorical encoding:
     - One-Hot Encoding for `Month` (9 columns) and `VisitorType` (2 columns)
     - Binary encoding for `Weekend` and `Revenue`
   - ✅ Feature scaling using `StandardScaler`
   - ✅ Train-test split (80/20) with stratification
   - ✅ SMOTE applied on training set only:
     - Training samples increased from 9,864 to 16,676
     - Perfect class balance achieved (8,338 each class)

2. **Model Training with MLFlow** (`code/train_models.py`)
   - ✅ MLFlow experiment tracking configured
   - ✅ Trained 4 models with full metric logging:
     - Logistic Regression (Baseline)
     - Decision Tree (Baseline)
     - Random Forest (Ensemble)
     - XGBoost (Ensemble)
   - ✅ All models saved to `models/` directory
   - ✅ Confusion matrices generated for each model

3. **Model Performance Comparison**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.8528 | 0.5173 | 0.7435 | 0.6101 | 0.8978 |
| **Decision Tree** | 0.8500 | 0.5112 | 0.7199 | 0.5978 | 0.8532 |
| **Random Forest** | 0.8836 | 0.6058 | 0.7120 | 0.6546 | 0.9192 |
| **XGBoost** | **0.8933** | **0.6537** | 0.6623 | **0.6580** | **0.9280** |

4. **Best Model Selection: 🏆 XGBoost**
   - **F1-Score**: 0.6580 (best balance of precision and recall)
   - **ROC-AUC**: 0.9280 (excellent class separation)
   - **Accuracy**: 89.33%
   - **Precision**: 65.37% (fewer false positives)
   - **Recall**: 66.23% (good detection of purchases)

### 📊 MLFlow Experiment Tracking

- **MLFlow UI**: Run `mlflow ui` and navigate to `http://localhost:5000`
- **Tracking**: All experiments logged with parameters, metrics, and artifacts
- **Artifacts**: Models, confusion matrices, and performance metrics stored
- **Database**: `mlflow.db` contains complete experiment history

---

### ✅ Week 3: Completed Tasks

1. **FastAPI Environment Setup** (`api/`)
   - ✅ Created `api/` package with proper module structure
   - ✅ Pydantic v2 schemas for strict input validation
   - ✅ Model loader with preprocessing pipeline mirroring training
   - ✅ Added dependencies: `fastapi`, `uvicorn`, `pydantic`

2. **Endpoints Implemented** (`api/main.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check – confirms API is alive |
| `GET` | `/model-info` | Model name, description & Week 2 metrics |
| `POST` | `/predict` | Single session → purchase prediction |
| `POST` | `/predict-batch` | Multiple sessions → list of predictions |

3. **Swagger UI Testing** (`http://localhost:8000/docs`)
   - ✅ All 4 endpoints tested and confirmed working
   - ✅ Interactive Swagger UI auto-generated by FastAPI

### 🔌 How to Run the API

```bash
# From project root
uvicorn api.main:app --reload
```

Then open:
- **Swagger UI** → `http://localhost:8000/docs` (interactive testing)
- **ReDoc** → `http://localhost:8000/redoc` (clean docs)

### 📡 Example API Usage

**POST `/predict`** – Single session:
```json
// Request body
{
  "Administrative": 0, "Administrative_Duration": 0.0,
  "Informational": 0, "Informational_Duration": 0.0,
  "ProductRelated": 35, "ProductRelated_Duration": 2500.0,
  "BounceRates": 0.01, "ExitRates": 0.03, "PageValues": 25.4,
  "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
  "Browser": 2, "Region": 1, "TrafficType": 2,
  "VisitorType": "Returning_Visitor", "Weekend": false
}

// Response
{
  "prediction": 1,
  "label": "Purchase",
  "purchase_probability": 0.8849,
  "no_purchase_probability": 0.1151
}
```

---

### Week 1-2 (Completed)
- **Python 3.x**, **Pandas**, **NumPy** - Data processing
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - ML algorithms and preprocessing
- **Imbalanced-learn** - SMOTE for class balancing
- **XGBoost** - Best performing model
- **MLflow** - Experiment tracking
- **Joblib** - Model persistence

### Week 3 (Completed)
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **Pydantic v2** - Input validation

### Future Weeks
- **Docker** - Containerization
- **React** - Frontend web interface

---

## 📈 Next Steps (Week 4+)

1. **Docker Containerization**
   - Dockerfile for the FastAPI app
   - `docker-compose.yml` for full stack deployment

2. **React Frontend**
   - Web form for submitting shopper sessions
   - Prediction results dashboard

---

## 👨‍💻 Author

**Course**: Python for Data Science – Guided Machine Learning  
**Institution**: University Course Project  
**Academic Year**: 2025-2026

---

## 📚 References

- Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2019). [Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks](https://link.springer.com/article/10.1007/s00521-018-3523-0)
- UCI Machine Learning Repository: [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

---

**Last Updated**: Week 3 - February 2026

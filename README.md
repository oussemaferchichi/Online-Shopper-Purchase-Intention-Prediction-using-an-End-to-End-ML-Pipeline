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
│
├── data/                           # Dataset and visualizations
│   ├── online_shoppers_intention.csv
│   └── plots/                      # EDA visualizations (generated from notebook)
│
├── notebooks/                      # Jupyter notebooks
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── frontend/                       # Future: React-based web interface
│
└── README.md                       # Project documentation
```

---

## 🚀 Week 1 Progress

### ✅ Completed Tasks

1. **Project Initialization**
   - Created organized directory structure
   - Set up data and visualization folders

2. **Exploratory Data Analysis** (`notebooks/eda.ipynb`)
   - Loaded and inspected dataset (shape, columns, data types)
   - Analyzed missing values
   - Visualized class distribution (Revenue)
   - Analyzed purchase patterns by:
     - Visitor Type (New, Returning, Other)
     - Month
   - Explored distributions of key numerical features:
     - PageValues
     - BounceRates
     - ExitRates
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

## 🛠️ Planned Technologies

### Week 1 (Current)
- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive analysis

### Future Weeks
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **Imbalanced-learn (imblearn)** - SMOTE for handling class imbalance
- **MLflow** - Experiment tracking and model registry
- **FastAPI** - REST API for model serving
- **Docker** - Containerization for deployment
- **React** - Frontend web interface

---

## 📈 Next Steps (Week 2+)

1. **Data Preprocessing**
   - Implement missing value imputation
   - Encode categorical variables
   - Scale numerical features
   - Apply train-test split
   - Apply SMOTE for class balancing

2. **Model Development**
   - Train baseline models (Logistic Regression, Decision Trees)
   - Experiment with ensemble methods (Random Forest, XGBoost)
   - Hyperparameter tuning
   - Model evaluation with appropriate metrics for imbalanced data

3. **MLOps & Deployment**
   - Track experiments with MLflow
   - Build FastAPI endpoints
   - Create Docker containers
   - Develop React frontend

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

**Last Updated**: Week 1 - February 2026

"""
Data Preprocessing Pipeline for Online Shopper Purchase Intention Prediction

This module handles:
- Missing value imputation
- Categorical encoding
- Feature scaling
- Train-test split
- SMOTE application for class balancing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_data(filepath):
    """Load the dataset from CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def check_missing_values(df):
    """Check for missing values in the dataset."""
    print("\n=== Missing Values Analysis ===")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found in the dataset.")
    else:
        print("Missing values per column:")
        print(missing[missing > 0])
    return missing


def encode_categorical_features(df):
    """
    Encode categorical features:
    - One-Hot Encoding for Month, VisitorType
    - Label Encoding for Weekend (already boolean)
    - Revenue is the target variable
    """
    print("\n=== Encoding Categorical Features ===")
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    # One-Hot Encoding for Month
    month_dummies = pd.get_dummies(df_encoded['Month'], prefix='Month', drop_first=True)
    df_encoded = pd.concat([df_encoded, month_dummies], axis=1)
    df_encoded.drop('Month', axis=1, inplace=True)
    print(f"Month encoded: {len(month_dummies.columns)} new columns created")
    
    # One-Hot Encoding for VisitorType
    visitor_dummies = pd.get_dummies(df_encoded['VisitorType'], prefix='VisitorType', drop_first=True)
    df_encoded = pd.concat([df_encoded, visitor_dummies], axis=1)
    df_encoded.drop('VisitorType', axis=1, inplace=True)
    print(f"VisitorType encoded: {len(visitor_dummies.columns)} new columns created")
    
    # Weekend is already boolean (True/False), convert to 1/0
    df_encoded['Weekend'] = df_encoded['Weekend'].astype(int)
    
    # Revenue (target) to 1/0
    df_encoded['Revenue'] = df_encoded['Revenue'].astype(int)
    
    print(f"Final encoded dataset shape: {df_encoded.shape}")
    return df_encoded


def split_features_target(df):
    """Split dataset into features (X) and target (y)."""
    print("\n=== Splitting Features and Target ===")
    
    # Separate features and target
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    Fit on training data only to prevent data leakage.
    """
    print("\n=== Scaling Features ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"Features scaled successfully")
    
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to balance the training dataset.
    Only applied to training data to prevent data leakage.
    """
    print("\n=== Applying SMOTE for Class Balancing ===")
    print(f"Before SMOTE - Class distribution:\n{y_train.value_counts()}")
    
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE - Class distribution:\n{pd.Series(y_train_balanced).value_counts()}")
    print(f"Training set increased from {len(y_train)} to {len(y_train_balanced)} samples")
    
    return X_train_balanced, y_train_balanced


def preprocess_pipeline(data_path, test_size=0.2, random_state=42, apply_smote_flag=True):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    test_size : float
        Proportion of dataset for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    apply_smote_flag : bool
        Whether to apply SMOTE for balancing (default: True)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Load data
    df = load_data(data_path)
    
    # Check missing values
    check_missing_values(df)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df)
    
    # Split features and target
    X, y = split_features_target(df_encoded)
    
    # Train-test split with stratification
    print("\n=== Train-Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Apply SMOTE if requested
    if apply_smote_flag:
        X_train_final, y_train_final = apply_smote(X_train_scaled, y_train, random_state)
    else:
        X_train_final, y_train_final = X_train_scaled, y_train
    
    # Store feature names
    feature_names = X_train.columns.tolist()
    
    print("\n=== Preprocessing Complete ===")
    print(f"Final training set: {X_train_final.shape}")
    print(f"Final test set: {X_test_scaled.shape}")
    
    return X_train_final, X_test_scaled, y_train_final, y_test, scaler, feature_names


def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, feature_names, output_dir='data'):
    """Save preprocessed data and scaler for later use."""
    print("\n=== Saving Preprocessed Data ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as joblib files
    joblib.dump(X_train, os.path.join(output_dir, 'X_train.pkl'))
    joblib.dump(X_test, os.path.join(output_dir, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))
    
    print(f"Preprocessed data saved to {output_dir}/")


if __name__ == "__main__":
    # Path to the dataset
    DATA_PATH = "data/online_shoppers_intention.csv"
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(
        DATA_PATH, 
        test_size=0.2, 
        random_state=42, 
        apply_smote_flag=True
    )
    
    # Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, feature_names)
    
    print("\n✅ Preprocessing pipeline completed successfully!")

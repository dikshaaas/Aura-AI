import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

def main():
    print("Initializing model training...")
    
    features_csv_path = os.path.join("data", "processed", "features.csv")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(features_csv_path):
        print(f"Error: {features_csv_path} not found! Please run extract_features.py first.")
        return
        
    # Load dataset
    df = pd.read_csv(features_csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Split into features X and target y
    X = df.drop(columns=['accent', 'file'])
    y = df['accent']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Create SVM Pipeline (Scaler + SVC)
    # probability=True is required to output probability distributions for the frontend
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(C=1, gamma=0.1, kernel='rbf', probability=True, random_state=42))
    ])
    
    print("Training Optimized SVM classifier...")
    svm_pipeline.fit(X_train, y_train)
    
    # Save train/test split so evaluation script can reuse them if needed, or it can load features.csv and split with same random_state
    # Save the trained model pipeline
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(svm_pipeline, model_path)
    print(f"Saved SVM model pipeline to {model_path}")
    
    # Also train Random Forest just to compare or show we did
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(class_weight='balanced', max_depth=20, n_estimators=200, random_state=42))
    ])
    print("Training Random Forest classifier for comparison...")
    rf_pipeline.fit(X_train, y_train)
    
    rf_model_path = os.path.join(models_dir, "rf_model.joblib")
    joblib.dump(rf_pipeline, rf_model_path)
    print(f"Saved Random Forest model pipeline to {rf_model_path}")
    
    print("Model training completed successfully.")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("Initializing model evaluation...")
    
    features_csv_path = os.path.join("data", "processed", "features.csv")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    if not os.path.exists(features_csv_path):
        print(f"Error: {features_csv_path} not found!")
        return
        
    # Load dataset
    df = pd.read_csv(features_csv_path)
    
    # Split into features X and target y
    X = df.drop(columns=['accent', 'file'])
    y = df['accent']
    
    # Split into train and test sets (must match train_model.py split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load models
    svm_path = os.path.join("models", "best_model.joblib")
    rf_path = os.path.join("models", "rf_model.joblib")
    
    if not os.path.exists(svm_path) or not os.path.exists(rf_path):
        print("Error: Models not found! Run train_model.py first.")
        return
        
    svm_model = joblib.load(svm_path)
    rf_model = joblib.load(rf_path)
    
    # Predict
    y_pred_svm = svm_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    
    # Metrics SVM
    acc_svm = accuracy_score(y_test, y_pred_svm) * 100
    prec_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
    rec_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
    
    # Metrics RF
    acc_rf = accuracy_score(y_test, y_pred_rf) * 100
    prec_rf = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
    
    # Find confused pairs for SVM
    conf_matrix = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)
    classes = list(svm_model.classes_)
    
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and conf_matrix[i, j] >= 2:
                confused_pairs.append({
                    'actual': classes[i],
                    'predicted': classes[j],
                    'count': conf_matrix[i, j]
                })
                
    # Sort confused pairs by count descending
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    # Generate model_comparison_report.txt content
    report_content = f"""MODEL COMPARISON REPORT
==================================================
Metric          | Optimized SVM   | Random Forest  
--------------------------------------------------
Accuracy        | {acc_svm:.2f}         % | {acc_rf:.2f}         %
Precision       | {prec_svm:.4f}          | {prec_rf:.4f}         
Recall          | {rec_svm:.4f}          | {rec_rf:.4f}         
F1-score        | {f1_svm:.4f}          | {f1_rf:.4f}         
==================================================

Best SVM Parameters: {{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}}
Best RF Parameters: {{'class_weight': 'balanced', 'max_depth': 20, 'n_estimators': 200}}
Selected Best Model: Optimized SVM

Commonly Confused Accent Pairs (count >= 2):
"""
    for pair in confused_pairs:
        report_content += f" - {pair['actual']} misclassified as {pair['predicted']}: {pair['count']} occurrences\n"
        
    report_path = os.path.join(reports_dir, "model_comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Saved comparison report to {report_path}")
    
    # Plot model comparison metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    svm_vals = [acc_svm / 100, prec_svm, rec_svm, f1_svm]
    rf_vals = [acc_rf / 100, prec_rf, rec_rf, f1_rf]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, svm_vals, width, label='Optimized SVM', color='#00b4d8')
    rects2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#7209b7')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    fig.tight_layout()
    plot_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved performance plot to {plot_path}")
    
if __name__ == "__main__":
    main()

"""
train_model.py
Trains and compares multiple ML models on the Breast Cancer Wisconsin dataset
to predict whether a tumor is malignant or benign based on cell measurements.
Saves the best model + scaler for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

# ---- Load data ----
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant, 1 = benign

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Scale features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Train multiple models and compare ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
    }
    trained_models[name] = model
    print(f"{name}: {results[name]}")

# ---- Pick best model by F1 score ----
best_model_name = max(results, key=lambda k: results[k]["f1_score"])
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---- Feature importance (Random Forest specifically, for the app's chart) ----
rf_model = trained_models["Random Forest"]
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# ---- Save everything the app needs ----
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("results.json", "w") as f:
    json.dump({
        "results": results,
        "best_model": best_model_name,
        "feature_names": list(X.columns),
        "feature_importance": feature_importance.to_dict(),
        "feature_ranges": {
            col: {"min": float(X[col].min()), "max": float(X[col].max()), "mean": float(X[col].mean())}
            for col in X.columns
        }
    }, f, indent=2)

# Save a sample of the raw data for the EDA tab in the app
X_with_target = X.copy()
X_with_target["diagnosis"] = ["Malignant" if v == 0 else "Benign" for v in y]
X_with_target.to_csv("data_sample.csv", index=False)

print("\nSaved: model.pkl, scaler.pkl, results.json, data_sample.csv")

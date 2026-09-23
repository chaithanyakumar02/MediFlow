import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

data = pd.read_csv("diabetes_data.csv")

print("Dataset loaded successfully")
print("Shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())


# ---------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------

# These columns cannot realistically have value 0.
# In this dataset, 0 is generally used to represent missing values.
zero_invalid_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in zero_invalid_columns:
    data[col] = data[col].replace(0, np.nan)

# Fill missing values using median
for col in zero_invalid_columns:
    data[col] = data[col].fillna(data[col].median())


# ---------------------------------------------------------
# 3. FEATURES AND TARGET
# ---------------------------------------------------------

feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

X = data[feature_names]
y = data["Outcome"]


# ---------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 5. FEATURE SCALING
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 6. MODELS
# ---------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}


# ---------------------------------------------------------
# 7. TRAIN AND EVALUATE MODELS
# ---------------------------------------------------------

results = {}

best_model = None
best_model_name = None
best_f1 = -1


for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# ---------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ---------------------------------------------------------

# Train Random Forest separately for feature importance
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

importance_values = rf.feature_importances_

feature_importance = {
    feature: float(importance)
    for feature, importance in zip(feature_names, importance_values)
}

# Sort highest importance first
feature_importance = dict(
    sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True
    )
)


# ---------------------------------------------------------
# 9. FEATURE RANGES FOR STREAMLIT SLIDERS
# ---------------------------------------------------------

feature_ranges = {}

for feature in feature_names:
    feature_ranges[feature] = {
        "min": float(X[feature].min()),
        "max": float(X[feature].max()),
        "mean": float(X[feature].mean())
    }


# ---------------------------------------------------------
# 10. SAVE BEST MODEL
# ---------------------------------------------------------

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# ---------------------------------------------------------
# 11. SAVE RESULTS
# ---------------------------------------------------------

output = {
    "best_model": best_model_name,
    "feature_names": feature_names,
    "results": results,
    "feature_importance": feature_importance,
    "feature_ranges": feature_ranges
}

with open("diabetes_results.json", "w") as f:
    json.dump(output, f, indent=4)


# ---------------------------------------------------------
# 12. SAVE CLEANED DATA
# ---------------------------------------------------------

data.to_csv("diabetes_data_cleaned.csv", index=False)


# ---------------------------------------------------------
# 13. FINAL OUTPUT
# ---------------------------------------------------------

print("\n-----------------------------------------")
print("Training completed successfully")
print("-----------------------------------------")

print(f"\nBest Model: {best_model_name}")
print(f"Best F1 Score: {best_f1:.4f}")

print("\nFiles generated:")
print("diabetes_model.pkl")
print("diabetes_scaler.pkl")
print("diabetes_results.json")
print("diabetes_data_cleaned.csv")
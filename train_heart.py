import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.inspection import permutation_importance


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

data = pd.read_csv("heart_data.csv")

print("Dataset loaded successfully")
print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nTarget Distribution:")
print(data["condition"].value_counts())


# ---------------------------------------------------------
# 2. FEATURES
# ---------------------------------------------------------

feature_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


numeric_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


X = data[feature_names]
y = data["condition"]


# ---------------------------------------------------------
# 3. TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 4. PREPROCESSOR
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# 5. MODELS
# ---------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}


# ---------------------------------------------------------
# 6. TRAIN
# ---------------------------------------------------------

results = {}

trained_models = {}

best_model = None
best_model_name = None
best_f1 = -1


for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    results[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }


    trained_models[name] = pipeline


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    if f1 > best_f1:

        best_f1 = f1

        best_model = pipeline

        best_model_name = name


# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------

# Use Random Forest pipeline with permutation importance.
# This gives importance for the ORIGINAL input columns.

rf_pipeline = trained_models[
    "Random Forest"
]


permutation = permutation_importance(
    rf_pipeline,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=20,
    random_state=42
)


feature_importance = {

    feature: float(importance)

    for feature, importance in zip(
        feature_names,
        permutation.importances_mean
    )
}


feature_importance = dict(
    sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True
    )
)


# ---------------------------------------------------------
# 8. FEATURE RANGES
# ---------------------------------------------------------

feature_ranges = {}


for feature in feature_names:

    feature_ranges[feature] = {

        "min": float(
            X[feature].min()
        ),

        "max": float(
            X[feature].max()
        ),

        "mean": float(
            X[feature].mean()
        )
    }


# ---------------------------------------------------------
# 9. SAVE MODEL
# ---------------------------------------------------------

with open(
    "heart_model.pkl",
    "wb"
) as f:

    pickle.dump(
        best_model,
        f
    )


# ---------------------------------------------------------
# 10. SAVE RESULTS
# ---------------------------------------------------------

output = {

    "best_model":
        best_model_name,

    "feature_names":
        feature_names,

    "results":
        results,

    "feature_importance":
        feature_importance,

    "feature_ranges":
        feature_ranges
}


with open(
    "heart_results.json",
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


# ---------------------------------------------------------
# 11. SAVE DATA
# ---------------------------------------------------------

data.to_csv(
    "heart_data_cleaned.csv",
    index=False
)


# ---------------------------------------------------------
# 12. FINAL OUTPUT
# ---------------------------------------------------------

print("\n-----------------------------------------")
print("Training completed successfully")
print("-----------------------------------------")


print(
    f"\nBest Model: {best_model_name}"
)


print(
    f"Best F1 Score: {best_f1:.4f}"
)


print("\nFiles generated:")

print(
    "heart_model.pkl"
)

print(
    "heart_results.json"
)

print(
    "heart_data_cleaned.csv"
)
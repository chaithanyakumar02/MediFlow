# 🏥 MediFlow

### Explainable Multi-Disease Risk Prediction System

MediFlow is a machine-learning based healthcare analytics application that predicts the risk of multiple diseases using patient and clinical features.

The application currently supports:

- 🎗️ Breast Cancer Prediction
- 🩸 Diabetes Risk Prediction
- ❤️ Heart Disease Risk Prediction

MediFlow also provides model comparison, prediction probabilities, feature importance, class distribution, and dataset insights through an interactive Streamlit dashboard.

> ⚠️ **Disclaimer:** MediFlow is an educational and research project. It is not intended to provide medical diagnosis or replace professional medical advice.

---

## 🚀 Features

### 🩺 Multi-Disease Prediction

MediFlow currently provides three independent disease prediction modules:

| Disease | Status |
|---|---|
| Breast Cancer | ✅ Available |
| Diabetes | ✅ Available |
| Heart Disease | ✅ Available |

Each disease module uses its own trained machine-learning model and preprocessing pipeline.

---

## 🤖 Machine Learning Models

For every disease module, multiple classification algorithms are trained and compared:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The model with the best F1-score is selected for prediction.

---

## ❤️ Heart Disease Model Performance

For the Heart Disease module, the following results were obtained:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 91.67% | 100% | 82.14% | 0.902 |
| Random Forest | 90.00% | 100% | 78.57% | 0.880 |
| SVM | 90.00% | 100% | 78.57% | 0.880 |

**Selected Model:** Logistic Regression

Categorical features are processed using One-Hot Encoding, while numerical features are standardized before training.

---

## 🩸 Diabetes Model Performance

The Diabetes module compares the same three classifiers.

During testing, SVM produced the highest F1-score:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 70.78% | 60.00% | 50.00% | 0.545 |
| Random Forest | 73.38% | 64.44% | 53.70% | 0.586 |
| SVM | 74.03% | 65.22% | 55.56% | 0.600 |

**Selected Model:** SVM

---

## 🎗️ Breast Cancer Prediction

The Breast Cancer module uses features from the Breast Cancer Wisconsin Diagnostic dataset.

To keep the user interface simple, the application exposes the most important diagnostic features while remaining features are automatically filled using dataset-average values.

The module predicts:

- Benign
- Malignant

and displays the model's prediction probability.

---

## 📊 Dashboard Features

MediFlow contains three main sections for every disease.

### 1. 🩺 Patient Assessment

Users provide patient or clinical parameters using:

- Sliders
- Dropdown menus
- Categorical selections

The application then displays:

- Predicted class
- Prediction confidence
- Probability visualization

---

### 2. 🤖 Model Performance

This section compares all trained machine-learning models using:

- Accuracy
- Precision
- Recall
- F1 Score

A visual bar chart is also provided to compare the models.

---

### 3. 📊 Data Insights

This section provides:

- Feature importance
- Target/class distribution
- Dataset preview
- Important predictor visualization

---

## 🧠 System Workflow

```text
Patient / Clinical Data
          │
          ▼
      Streamlit UI
          │
          ▼
   Data Preprocessing
          │
          ▼
 ┌─────────────────────┐
 │ Machine Learning    │
 │                     │
 │ Logistic Regression │
 │ Random Forest       │
 │ SVM                 │
 └─────────────────────┘
          │
          ▼
     Best Model
          │
          ▼
 Disease Risk Prediction
          │
          ▼
 Probability + Insights

"""
app.py
Streamlit dashboard for the Breast Cancer Risk Predictor.
Three tabs: Predict (interactive input -> live prediction),
            Model Comparison (accuracy/precision/recall/f1 across 3 models),
            Data Insights (EDA - feature importance + distributions)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="MediFlow",
    page_icon="🏥",
    layout="wide"
)
# ---- Load saved artifacts ----
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("results.json", "r") as f:
        results = json.load(f)
    data = pd.read_csv("data_sample.csv")
    return model, scaler, results, data

model, scaler, results, data = load_artifacts()
feature_names = results["feature_names"]

st.sidebar.title("🏥 MediFlow")

st.sidebar.markdown("---")

st.sidebar.success("✅ Active Module\n\nBreast Cancer")

st.sidebar.info(
    "🚧 Upcoming\n\n"
    "- Diabetes\n"
    "- Heart Disease"
)

st.sidebar.markdown("---")
st.subheader("Explainable Multi-Disease Risk Prediction System")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🩺 Active Module", "Breast Cancer")

with col2:
    st.metric("🚧 Upcoming", "2 Diseases")

with col3:
    st.metric("🤖 ML Models", "3")
st.caption(
    """
    MediFlow is an AI-powered clinical decision support system designed for
    disease risk prediction using Machine Learning.

    ✅ Available: Breast Cancer Prediction

    🚧 Coming Soon:
    • Heart Disease
    • Diabetes

    Educational project only. Not intended for clinical diagnosis.
    """
)

disease = st.sidebar.selectbox(
    "Select Disease",
    [
        "Breast Cancer",
        "Heart Disease (Coming Soon)",
        "Diabetes (Coming Soon)"
    ]
)

if disease != "Breast Cancer":
    st.info(f"{disease} module is currently under development.")
    st.stop()
tab1, tab2, tab3 = st.tabs([
    "🩺 Patient Assessment",
    "🤖 Model Performance",
    "📈 Data Insights"
])
# ---------------- TAB 1: PREDICT ----------------
with tab1:
    st.subheader("Breast Cancer Assessment")
    st.write("Adjust the sliders below (defaults set to dataset averages) and get a live prediction.")

    # Use the 6 most important features for a clean UI (rest default to mean)
    top_features = list(results["feature_importance"].keys())[:6]

    col1, col2, col3 = st.columns(3)
    user_input = {}
    cols = [col1, col2, col3]

    for i, feat in enumerate(top_features):
        rng = results["feature_ranges"][feat]
        with cols[i % 3]:
            user_input[feat] = st.slider(
                feat.replace("_", " ").title(),
                min_value=float(rng["min"]),
                max_value=float(rng["max"]),
                value=float(rng["mean"]),
                key=feat
            )

    # Fill remaining features with dataset mean
    full_input = []
    for feat in feature_names:
        if feat in user_input:
            full_input.append(user_input[feat])
        else:
            full_input.append(results["feature_ranges"][feat]["mean"])

    if st.button("🔍 Predict", type="primary"):
        X_input = np.array(full_input).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        st.divider()
        if pred == 1:
            st.success(f"### ✅ Prediction: Benign")
            st.write(f"Confidence: **{proba[1]*100:.1f}%**")
        else:
            st.error(f"### ⚠️ Prediction: Malignant")
            st.write(f"Confidence: **{proba[0]*100:.1f}%**")

        prob_df = pd.DataFrame({
            "Outcome": ["Malignant", "Benign"],
            "Probability": [proba[0], proba[1]]
        })
        st.bar_chart(prob_df.set_index("Outcome"))

        st.caption("""⚠️ Disclaimer:
                   This application is intended for educational and research purposes only 
                   and should not be used as a substitute for professional medical advice or diagnosis.""")

# ---------------- TAB 2: MODEL COMPARISON ----------------
with tab2:
    st.subheader("Model Performance Comparison")
    results_df = pd.DataFrame(results["results"]).T
    results_df = results_df[["accuracy", "precision", "recall", "f1_score"]]
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Metrics")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.info(f"**{results['best_model']}** was selected as the production model based on highest F1-score, "
            "which balances precision and recall — important in medical diagnosis where both false "
            "positives and false negatives carry real cost.")

# ---------------- TAB 3: DATA INSIGHTS ----------------
with tab3:
    st.subheader("Feature Importance (Random Forest)")
    importance = pd.Series(results["feature_importance"]).sort_values(ascending=True).tail(10)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax2, color="teal")
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Top 10 Most Important Features")
    st.pyplot(fig2)

    st.subheader("Class Distribution")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    data["diagnosis"].value_counts().plot(kind="bar", ax=ax3, color=["salmon", "seagreen"])
    ax3.set_title("Malignant vs Benign Cases in Dataset")
    plt.xticks(rotation=0)
    st.pyplot(fig3)

    st.subheader("Raw Data Sample")
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown("---")
    st.caption(
        "🏥 MediFlow v1.0 | AI-powered Clinical Decision Support System | Educational Use Only"
    )

st.divider()
st.caption("Built with Python, scikit-learn, and Streamlit | Dataset: UCI Breast Cancer Wisconsin (Diagnostic)")

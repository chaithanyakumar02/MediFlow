import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="MediFlow",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# UI THEME
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1280px;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(128, 128, 128, 0.04);
        }

        .hero {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 22px;
            padding: 24px 26px;
            margin-bottom: 1rem;
            background:
                radial-gradient(circle at top right, rgba(79, 70, 229, 0.13), transparent 36%),
                radial-gradient(circle at bottom left, rgba(16, 185, 129, 0.10), transparent 32%),
                rgba(128, 128, 128, 0.035);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
            line-height: 1.1;
        }

        .hero p {
            margin: 0.55rem 0 0 0;
            opacity: 0.78;
            font-size: 1rem;
        }

        .badge-row {
            margin-top: 0.9rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .badge {
            display: inline-block;
            padding: 0.32rem 0.66rem;
            border-radius: 999px;
            border: 1px solid rgba(128, 128, 128, 0.20);
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(128, 128, 128, 0.06);
        }

        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            opacity: 0.62;
            margin-bottom: 0.15rem;
        }

        .result-card {
            border-radius: 18px;
            padding: 18px 20px;
            margin: 0.65rem 0 0.8rem 0;
            border: 1px solid rgba(128, 128, 128, 0.18);
            background: rgba(128, 128, 128, 0.045);
        }

        .result-card h3 {
            margin: 0 0 0.35rem 0;
        }

        .result-card p {
            margin: 0;
            opacity: 0.82;
        }

        div[data-testid="stTabs"] button {
            font-weight: 650;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
            min-height: 44px;
        }

        [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOADERS
# ---------------------------------------------------------

@st.cache_resource
def load_breast_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("results.json", "r") as f:
        results = json.load(f)

    data = pd.read_csv("data_sample.csv")

    return model, scaler, results, data


@st.cache_resource
def load_diabetes_artifacts():
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("diabetes_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("diabetes_results.json", "r") as f:
        results = json.load(f)

    data = pd.read_csv("diabetes_data_cleaned.csv")

    return model, scaler, results, data


@st.cache_resource
def rebuild_heart_model():
    """
    Rebuilds the heart-disease pipeline from heart_data.csv.
    This is used only if heart_model.pkl cannot be unpickled in deployment.
    """
    data = pd.read_csv("heart_data.csv")

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

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    return model


@st.cache_resource
def load_heart_artifacts():
    with open("heart_results.json", "r") as f:
        results = json.load(f)

    data = pd.read_csv("heart_data_cleaned.csv")

    used_fallback = False

    try:
        with open("heart_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = rebuild_heart_model()
        used_fallback = True

    return model, results, data, used_fallback


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def pretty_feature(name):
    labels = {
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "thalach": "Maximum Heart Rate",
        "exang": "Exercise-Induced Angina",
        "oldpeak": "ST Depression",
        "cp": "Chest Pain Type",
        "restecg": "Resting ECG",
        "fbs": "Fasting Blood Sugar",
        "ca": "Major Vessels",
        "thal": "Thalassemia",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "BloodPressure": "Blood Pressure",
        "SkinThickness": "Skin Thickness",
    }
    return labels.get(name, name.replace("_", " ").title())


def show_result(title, confidence, detail):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="section-label">Prediction Result</div>
            <h3>{title}</h3>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(confidence))
    st.caption(f"Model confidence: {confidence * 100:.1f}%")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown("## 🏥 MediFlow")
st.sidebar.caption("Multi-disease ML risk prediction")
st.sidebar.markdown("---")

disease = st.sidebar.selectbox(
    "Choose assessment",
    ["Breast Cancer", "Diabetes", "Heart Disease"]
)

st.sidebar.success("3 modules active")
st.sidebar.caption("Breast Cancer • Diabetes • Heart Disease")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.caption(
    "MediFlow compares multiple machine-learning models and provides "
    "prediction, model performance, and dataset insights in one dashboard."
)

st.sidebar.warning(
    "Educational project only. Predictions are not a medical diagnosis."
)


# ---------------------------------------------------------
# LAZY LOAD SELECTED MODULE ONLY
# ---------------------------------------------------------

heart_fallback = False

try:
    if disease == "Breast Cancer":
        model, scaler, results, data = load_breast_artifacts()

        feature_names = results["feature_names"]
        target_column = "diagnosis"
        assessment_title = "Breast Cancer Assessment"
        dataset_name = "UCI Breast Cancer Wisconsin (Diagnostic)"
        module_icon = "🎗️"

    elif disease == "Diabetes":
        model, scaler, results, data = load_diabetes_artifacts()

        feature_names = results["feature_names"]
        target_column = "Outcome"
        assessment_title = "Diabetes Risk Assessment"
        dataset_name = "Pima Indians Diabetes Dataset"
        module_icon = "🩸"

    else:
        model, results, data, heart_fallback = load_heart_artifacts()
        scaler = None

        feature_names = results["feature_names"]
        target_column = "condition"
        assessment_title = "Heart Disease Risk Assessment"
        dataset_name = "Cleveland Heart Disease Dataset"
        module_icon = "❤️"

except FileNotFoundError as e:
    st.error(f"Required project file is missing: {e.filename}")
    st.stop()

except Exception as e:
    st.error("The selected module could not be loaded.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
        <div class="section-label">MediFlow v2.0</div>
        <h1>{module_icon} {assessment_title}</h1>
        <p>
            Explainable machine-learning risk assessment with model comparison
            and dataset insights.
        </p>
        <div class="badge-row">
            <span class="badge">3 disease modules</span>
            <span class="badge">3 ML models compared</span>
            <span class="badge">{results["best_model"]} selected</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if disease == "Heart Disease" and heart_fallback:
    st.info(
        "Heart Disease model was rebuilt from the project dataset because the "
        "saved model artifact was not compatible with the current deployment environment."
    )


# ---------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------

metrics_df = pd.DataFrame(results["results"]).T
best_row = metrics_df.loc[results["best_model"]]

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Best Model", results["best_model"])

with m2:
    st.metric("Accuracy", f"{best_row['accuracy'] * 100:.1f}%")

with m3:
    st.metric("Recall", f"{best_row['recall'] * 100:.1f}%")

with m4:
    st.metric("F1 Score", f"{best_row['f1_score']:.3f}")


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["🩺 Patient Assessment", "🤖 Model Performance", "📊 Data Insights"]
)


# =========================================================
# TAB 1 — PATIENT ASSESSMENT
# =========================================================

with tab1:
    st.markdown("### Patient inputs")
    st.caption(
        "Enter the available measurements below. "
        "The model will return a risk classification and probability."
    )

    user_input = {}

    # -----------------------------------------------------
    # BREAST CANCER
    # -----------------------------------------------------

    if disease == "Breast Cancer":
        with st.container(border=True):
            st.markdown("#### Key diagnostic features")
            st.caption(
                "The interface exposes the six most important features. "
                "Remaining features use dataset-average values."
            )

            top_features = list(results["feature_importance"].keys())[:6]
            cols = st.columns(3)

            for i, feature in enumerate(top_features):
                feature_range = results["feature_ranges"][feature]

                with cols[i % 3]:
                    user_input[feature] = st.slider(
                        pretty_feature(feature),
                        min_value=float(feature_range["min"]),
                        max_value=float(feature_range["max"]),
                        value=float(feature_range["mean"]),
                        key=f"breast_{feature}"
                    )

    # -----------------------------------------------------
    # DIABETES
    # -----------------------------------------------------

    elif disease == "Diabetes":
        with st.container(border=True):
            st.markdown("#### Metabolic and patient profile")
            st.caption(
                "Values are constrained to the range observed in the training dataset."
            )

            cols = st.columns(3)

            integer_features = {
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "Age"
            }

            for i, feature in enumerate(feature_names):
                feature_range = results["feature_ranges"][feature]

                with cols[i % 3]:
                    if feature in integer_features:
                        user_input[feature] = st.slider(
                            pretty_feature(feature),
                            min_value=int(feature_range["min"]),
                            max_value=int(feature_range["max"]),
                            value=int(round(feature_range["mean"])),
                            key=f"diabetes_{feature}"
                        )
                    else:
                        step = 0.01 if feature == "DiabetesPedigreeFunction" else 0.1

                        user_input[feature] = st.slider(
                            pretty_feature(feature),
                            min_value=float(feature_range["min"]),
                            max_value=float(feature_range["max"]),
                            value=float(feature_range["mean"]),
                            step=step,
                            key=f"diabetes_{feature}"
                        )

    # -----------------------------------------------------
    # HEART DISEASE
    # -----------------------------------------------------

    else:
        with st.container(border=True):
            st.markdown("#### Patient profile")

            c1, c2, c3 = st.columns(3)

            with c1:
                rng = results["feature_ranges"]["age"]
                user_input["age"] = st.slider(
                    "Age",
                    min_value=int(rng["min"]),
                    max_value=int(rng["max"]),
                    value=int(round(rng["mean"])),
                    key="heart_age"
                )

            with c2:
                mapping = {"Female": 0, "Male": 1}
                choice = st.selectbox("Sex", list(mapping.keys()), key="heart_sex")
                user_input["sex"] = mapping[choice]

            with c3:
                mapping = {
                    "Typical Angina": 0,
                    "Atypical Angina": 1,
                    "Non-anginal Pain": 2,
                    "Asymptomatic": 3
                }
                choice = st.selectbox(
                    "Chest Pain Type",
                    list(mapping.keys()),
                    key="heart_cp"
                )
                user_input["cp"] = mapping[choice]

        with st.container(border=True):
            st.markdown("#### Clinical measurements")

            c1, c2, c3 = st.columns(3)

            with c1:
                rng = results["feature_ranges"]["trestbps"]
                user_input["trestbps"] = st.slider(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=int(rng["min"]),
                    max_value=int(rng["max"]),
                    value=int(round(rng["mean"])),
                    key="heart_trestbps"
                )

                mapping = {"No": 0, "Yes": 1}
                choice = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dL",
                    list(mapping.keys()),
                    key="heart_fbs"
                )
                user_input["fbs"] = mapping[choice]

            with c2:
                rng = results["feature_ranges"]["chol"]
                user_input["chol"] = st.slider(
                    "Serum Cholesterol (mg/dL)",
                    min_value=int(rng["min"]),
                    max_value=int(rng["max"]),
                    value=int(round(rng["mean"])),
                    key="heart_chol"
                )

                mapping = {
                    "Normal": 0,
                    "ST-T Wave Abnormality": 1,
                    "Left Ventricular Hypertrophy": 2
                }
                choice = st.selectbox(
                    "Resting ECG",
                    list(mapping.keys()),
                    key="heart_restecg"
                )
                user_input["restecg"] = mapping[choice]

            with c3:
                rng = results["feature_ranges"]["thalach"]
                user_input["thalach"] = st.slider(
                    "Maximum Heart Rate Achieved",
                    min_value=int(rng["min"]),
                    max_value=int(rng["max"]),
                    value=int(round(rng["mean"])),
                    key="heart_thalach"
                )

                mapping = {"No": 0, "Yes": 1}
                choice = st.selectbox(
                    "Exercise-Induced Angina",
                    list(mapping.keys()),
                    key="heart_exang"
                )
                user_input["exang"] = mapping[choice]

        with st.container(border=True):
            st.markdown("#### Exercise and diagnostic findings")

            c1, c2, c3 = st.columns(3)

            with c1:
                rng = results["feature_ranges"]["oldpeak"]
                user_input["oldpeak"] = st.slider(
                    "ST Depression (Oldpeak)",
                    min_value=float(rng["min"]),
                    max_value=float(rng["max"]),
                    value=float(rng["mean"]),
                    step=0.1,
                    key="heart_oldpeak"
                )

                mapping = {
                    "Upsloping": 0,
                    "Flat": 1,
                    "Downsloping": 2
                }
                choice = st.selectbox(
                    "Peak Exercise ST Segment",
                    list(mapping.keys()),
                    key="heart_slope"
                )
                user_input["slope"] = mapping[choice]

            with c2:
                rng = results["feature_ranges"]["ca"]
                user_input["ca"] = st.selectbox(
                    "Major Vessels Colored by Fluoroscopy",
                    options=list(
                        range(
                            int(rng["min"]),
                            int(rng["max"]) + 1
                        )
                    ),
                    key="heart_ca"
                )

            with c3:
                mapping = {
                    "Normal": 0,
                    "Fixed Defect": 1,
                    "Reversible Defect": 2
                }
                choice = st.selectbox(
                    "Thalassemia",
                    list(mapping.keys()),
                    key="heart_thal"
                )
                user_input["thal"] = mapping[choice]


    # -----------------------------------------------------
    # BUILD INPUT
    # -----------------------------------------------------

    full_input = []

    for feature in feature_names:
        if feature in user_input:
            full_input.append(user_input[feature])
        else:
            full_input.append(results["feature_ranges"][feature]["mean"])


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    st.markdown("")

    if st.button(
        "Run Risk Assessment",
        type="primary",
        use_container_width=True,
        key=f"predict_{disease}"
    ):
        input_df = pd.DataFrame([full_input], columns=feature_names)

        if disease == "Heart Disease":
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
        else:
            X_scaled = scaler.transform(input_df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]

        st.markdown("---")

        if disease == "Breast Cancer":
            if prediction == 1:
                confidence = float(probability[1])
                show_result(
                    "✅ Prediction: Benign",
                    confidence,
                    "The model classified this input as benign."
                )
            else:
                confidence = float(probability[0])
                show_result(
                    "⚠️ Prediction: Malignant",
                    confidence,
                    "The model classified this input as malignant."
                )

            probability_df = pd.DataFrame(
                {
                    "Outcome": ["Malignant", "Benign"],
                    "Probability": [probability[0], probability[1]]
                }
            )

        elif disease == "Diabetes":
            if prediction == 1:
                confidence = float(probability[1])
                show_result(
                    "⚠️ Higher Diabetes Risk",
                    confidence,
                    "The model classified this input in the higher-risk class."
                )
            else:
                confidence = float(probability[0])
                show_result(
                    "✅ Lower Diabetes Risk",
                    confidence,
                    "The model classified this input in the lower-risk class."
                )

            probability_df = pd.DataFrame(
                {
                    "Outcome": ["Lower Risk", "Higher Risk"],
                    "Probability": [probability[0], probability[1]]
                }
            )

        else:
            if prediction == 1:
                confidence = float(probability[1])
                show_result(
                    "⚠️ Higher Heart Disease Risk",
                    confidence,
                    "The model classified this input in the higher-risk class."
                )
            else:
                confidence = float(probability[0])
                show_result(
                    "✅ Lower Heart Disease Risk",
                    confidence,
                    "The model classified this input in the lower-risk class."
                )

            probability_df = pd.DataFrame(
                {
                    "Outcome": ["Lower Risk", "Higher Risk"],
                    "Probability": [probability[0], probability[1]]
                }
            )

        with st.container(border=True):
            st.markdown("#### Prediction probability")
            st.bar_chart(probability_df.set_index("Outcome"))

        st.info(
            "This prediction is generated by a machine-learning model for "
            "educational and research purposes. It should not be used as a "
            "substitute for professional medical evaluation."
        )


# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================

with tab2:
    st.markdown("### Model comparison")
    st.caption(
        f"Three classifiers were evaluated for the {disease} module. "
        f"{results['best_model']} was selected using F1-score."
    )

    results_df = pd.DataFrame(results["results"]).T
    results_df = results_df[
        ["accuracy", "precision", "recall", "f1_score"]
    ]

    display_df = results_df.copy()
    display_df.columns = ["Accuracy", "Precision", "Recall", "F1 Score"]

    st.dataframe(
        display_df.style.format("{:.3f}").highlight_max(axis=0),
        use_container_width=True
    )

    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        results_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(f"{disease} — Model Comparison")
        ax.legend(loc="lower right")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.info(
        f"**Selected production model: {results['best_model']}**. "
        "F1-score balances precision and recall and is more informative than "
        "accuracy alone for binary medical classification."
    )


# =========================================================
# TAB 3 — DATA INSIGHTS
# =========================================================

with tab3:
    left, right = st.columns([1.35, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### Feature importance")

            importance = pd.Series(
                results["feature_importance"]
            ).sort_values(ascending=True).tail(10)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            importance.plot(kind="barh", ax=ax2)
            ax2.set_xlabel("Importance Score")
            ax2.set_title(f"Top Features — {disease}")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("#### Class distribution")

            distribution_data = data[target_column].copy()

            if disease == "Breast Cancer":
                distribution_data = distribution_data.replace(
                    {"M": "Malignant", "B": "Benign"}
                )
            elif disease == "Diabetes":
                distribution_data = distribution_data.replace(
                    {0: "No Diabetes", 1: "Diabetes"}
                )
            else:
                distribution_data = distribution_data.replace(
                    {0: "No Heart Disease", 1: "Heart Disease"}
                )

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            distribution_data.value_counts().plot(kind="bar", ax=ax3)
            ax3.set_ylabel("Records")
            ax3.set_title("Target Distribution")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

    st.markdown("### Dataset preview")
    st.caption(f"Source used by this module: {dataset_name}")

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")
st.caption(
    f"🏥 MediFlow v2.0 • {dataset_name} • "
    "Python • scikit-learn • Streamlit • Educational Use Only"
)

import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# =========================================================
# LOAD MODEL AND PREPROCESSOR
# =========================================================

preprocessor = joblib.load("heart_preprocessor.pkl")

model = xgb.XGBClassifier()
model.load_model("heart_xgb_model.json")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 17px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #ddd;
    margin-top: 20px;
}

.result-title {
    font-size: 28px;
    font-weight: 700;
}

.probability {
    font-size: 38px;
    font-weight: 700;
    margin: 10px 0;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An ML-powered system for estimating the likelihood of heart disease '
    'based on patient health information.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )


# =========================================================
# HEART INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🫀 Heart & Medical Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "asymptomatic",
            "atypical angina",
            "non-anginal",
            "typical angina"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=50,
        max_value=700,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [
            "lv hypertrophy",
            "normal",
            "st-t abnormality"
        ]
    )


with col2:

    thalch = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )

    slope = st.selectbox(
        "ST Segment Slope",
        [
            "downsloping",
            "flat",
            "upsloping"
        ]
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Convert Yes/No to Boolean values

    fbs_value = True if fbs == "Yes" else False
    exang_value = True if exang == "Yes" else False


    # Create input DataFrame

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs_value],
        "restecg": [restecg],
        "thalch": [thalch],
        "exang": [exang_value],
        "oldpeak": [oldpeak],
        "slope": [slope]
    })


    # Apply preprocessing

    processed_input = preprocessor.transform(input_data)


    # Make prediction

    prediction = model.predict(processed_input)[0]

    probability = (
        model.predict_proba(processed_input)[0][1] * 100
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.error(
            "⚠️ Higher likelihood of heart disease"
        )

        result_text = "Higher Likelihood"

    else:

        st.success(
            "🟢 Lower likelihood of heart disease"
        )

        result_text = "Lower Likelihood"


    # Result box

    st.markdown(
        f'<div class="result-box">'
        f'<div class="result-title">{result_text}</div>'
        f'<div class="probability">{probability:.2f}%</div>'
        f'<div>Estimated probability of heart disease</div>'
        f'</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.progress(int(probability))


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability < 30:

        risk_level = "🟢 Low Risk"

    elif probability < 60:

        risk_level = "🟠 Moderate Risk"

    else:

        risk_level = "🔴 Higher Risk"


    st.markdown(
        f"### Risk Level: {risk_level}"
    )


    # =====================================================
    # PATIENT SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">👤 Patient Summary</div>',
        unsafe_allow_html=True
    )


    summary = pd.DataFrame({

        "Parameter": [
            "Age",
            "Sex",
            "Chest Pain",
            "Resting BP",
            "Cholesterol",
            "Max Heart Rate",
            "Oldpeak"
        ],

        "Value": [
            age,
            sex,
            cp,
            trestbps,
            chol,
            thalch,
            oldpeak
        ]

    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.markdown(
        '<div class="section-title">🤖 Model Information</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "82.07%")
    col2.metric("Recall", "88.24%")
    col3.metric("F1 Score", "84.51%")
    col4.metric("ROC-AUC", "90.94%")


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "⚕️ This system is developed for educational and "
        "demonstration purposes. The prediction is generated "
        "by a machine learning model and should not be "
        "considered a medical diagnosis."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'Heart Disease Prediction System • Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)

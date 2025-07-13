import streamlit as st
import joblib
import pandas as pd

pipeline = joblib.load('./models/best_model_pipeline.pkl')

st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"] {
        background-color: #121212;  
        color: #E0E0E0;  
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    div[data-testid="stSidebar"] {
        background-color: #1F1F1F;
        color: #E0E0E0;
    }

    .css-10trblm {
        font-size: 3.2rem !important;
        font-weight: 900 !important;
        color: #D32F2F !important; 
        text-shadow: 2px 2px 6px rgba(211, 47, 47, 0.7);
        margin-bottom: 25px !important;
        font-family: 'Poppins', sans-serif;
    }

    /* تنسيق صناديق الإدخال */
    .stNumberInput input, .stSelectbox div[role="combobox"] {
        background-color: #222222 !important;
        color: #FFFFFF !important;
        border: 2px solid #B71C1C !important; 
        border-radius: 8px !important;
        box-shadow: 0 0 8px #B71C1CAA;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 8px !important;
    }

    .stSelectbox > div {
        border: 2px solid #B71C1C !important;
        border-radius: 8px !important;
        box-shadow: 0 0 8px #B71C1CAA;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    label, .css-1aumxhk, .css-1offfwp {
        color: #F5F5F5 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 4px;
    }

    .stButton>button {
        background-color: #B71C1C;
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.1rem;
        box-shadow: 0 0 10px #B71C1CAA;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #D32F2F;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Heart Disease Prediction ❤️")
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)", [0, 1])
thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest (Oldpeak)", min_value=0.0, max_value=10.0, value=0.0, format="%.2f")
ca = st.number_input("Number of Major Vessels Colored (0-3)", min_value=0, max_value=3, value=0)

cp_2_0 = st.selectbox("Atypical Angina(0 or 1)", [0, 1])
cp_3_0 = st.selectbox("Non-anginal Pain (0 or 1)", [0, 1])
cp_4_0 = st.selectbox("Asymptomatic (0 or 1)", [0, 1])

restecg_1_0 = st.selectbox("Having ST-T wave abnormality (0 or 1)", [0, 1])
restecg_2_0 = st.selectbox("Showing probable or definite left ventricula hypertrophy (0 or 1)", [0, 1])

slope_2_0 = st.selectbox("Flat slope (0 or 1)", [0, 1])
slope_3_0 = st.selectbox("Downsloping slope (0 or 1)", [0, 1])

thal_6_0 = st.selectbox("Fixed Defect (0 or 1)", [0, 1])
thal_7_0 = st.selectbox("Reversible Defect(0 or 1)", [0, 1])
thal_unknown = st.selectbox("Unknown Thalassemia Status (0 or 1)", [0, 1])

if st.button("Predict"):
    input_df = pd.DataFrame([[
        age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca,
        cp_2_0, cp_3_0, cp_4_0,
        restecg_1_0, restecg_2_0,
        slope_2_0, slope_3_0,
        thal_6_0, thal_7_0, thal_unknown
    ]], columns=[
        'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
        'cp_2.0', 'cp_3.0', 'cp_4.0',
        'restecg_1.0', 'restecg_2.0',
        'slope_2.0', 'slope_3.0',
        'thal_6.0', 'thal_7.0', 'thal_?'
    ])

    prediction = pipeline.predict(input_df)[0]

    if prediction == 0:
        st.success("✅ Prediction: No Heart Disease (Very Low Risk)")
    elif prediction == 1:
        st.info("ℹ️ Prediction: Mild Heart Disease (Low Risk)")
    elif prediction == 2:
        st.warning("⚠️ Prediction: Moderate Heart Disease (Medium Risk)")
    elif prediction == 3:
        st.error("❌ Prediction: Severe Heart Disease (High Risk)")
    elif prediction == 4:
        st.error("🚨 Prediction: Very Severe Heart Disease (Very High Risk)")
    else:
        st.info(f"🔎 Prediction: {prediction}")
st.markdown(
    """
    <div style="background-color:#800020; padding:15px; border-radius:10px; color:white; font-weight:bold; font-family: Arial, sans-serif;">
        <p>⚠️ Note: This prediction may not be 100% accurate. Please consult a healthcare professional for an official diagnosis.</p>
        <p>🙏 Wishing you good health and well-being.</p>
        <p style="margin-top:30px; font-size:16px;">🔖 Best regards,</p>
        <p style="font-family: 'Courier New', Courier, monospace; font-size:20px; letter-spacing:3px; margin-top:-10px;">Gehad Abdulaziz</p>
    </div>
    """,
    unsafe_allow_html=True
)

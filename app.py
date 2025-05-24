import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- File Paths ---
MODEL_PATH = "mental_health_model.pkl"
DATA_PATH = "dataset.csv"

# --- Train Model and Save ---
def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    for col in df.columns[:-1]:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    le = LabelEncoder()
    df['Disorder'] = le.fit_transform(df['Disorder'])
    X = df.drop(columns=['Disorder'])
    y = df['Disorder']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, le, list(X.columns)), MODEL_PATH)
    return model, le, list(X.columns)

# --- Load Model ---
def load_model():
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, tuple):
        if len(loaded) == 3:
            return loaded
        elif len(loaded) == 2:
            model, le = loaded
            feature_names = list(model.feature_names_in_)
            return model, le, feature_names
    raise ValueError("Model file format is incorrect.")

# --- Check and Load/Train Model ---
if not os.path.exists(MODEL_PATH):
    model, le, feature_names = train_and_save_model()
else:
    try:
        model, le, feature_names = load_model()
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()

# --- Streamlit App Config ---
st.set_page_config(page_title="AI Mental Health Diagnosis", page_icon="🧠", layout="wide")

# --- Page Header ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 3em;
        color: #5C4D7D;
        font-weight: bold;
        text-align: center;
    }
    .sub-title {
        text-align: center;
        font-size: 1.5em;
        color: #777;
        margin-bottom: 1em;
    }
    .note {
        font-style: italic;
        text-align: center;
        color: #999;
    }
    .stButton button {
        background-color: #6c63ff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧠 AI-Powered Mental Health Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Professional, AI-Driven Mental Health Screening Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="note">*Note: This tool is for educational and awareness purposes only.*</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4359/4359966.png", width=100)
    st.header("ℹ️ About This Tool")
    st.info("This AI assistant gives a preliminary mental health diagnosis based on your symptoms using machine learning.")
    st.warning("🛑 This tool does **not replace** a certified medical professional.")
    st.markdown("Created with ❤️ by Supriya Lahri")

# --- User Input Form ---
st.subheader("📝 Select Your Symptoms")
st.markdown("Toggle the symptoms that you're experiencing:")

with st.form("symptom_form"):
    cols = st.columns(3)
    user_input = {}

    for i, col in enumerate(feature_names):
        label = col.replace('_', ' ').capitalize()
        user_input[col] = cols[i % 3].toggle(label, value=False, key=f"symptom_{i}")

    submitted = st.form_submit_button("🔍 Diagnose", use_container_width=True)

# --- Prediction Section ---
if submitted:
    input_data = np.array([1 if user_input[col] else 0 for col in feature_names]).reshape(1, -1)
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    diagnosis = le.inverse_transform(prediction)[0]
    confidence = np.max(proba) * 100

    # Result Box
    st.markdown("### 🧾 Diagnosis Result")
    st.success(f"#### ✅ Mental Health Condition: **{diagnosis}**")
    st.progress(int(confidence), f"Model Confidence: {confidence:.2f}%")

    # Feature Importance
    st.markdown("#### 🔍 Important Symptoms Influencing the Prediction")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Symptom": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Symptom"))

    # Final Note
    st.markdown("---")
    st.warning("🔔 This is a preliminary assessment. If you have concerns, please consult a professional mental health expert.")


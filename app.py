
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



# Load dataset
df = pd.read_csv('dataset.csv')

# Convert 'yes'/'no' to binary (1/0)
for col in df.columns[:-1]:  # Exclude 'Disorder' column
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Encode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Disorder'] = le.fit_transform(df['Disorder'])

# Train model
X = df.drop(columns=['Disorder'])
y = df['Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump((model, le), "mental_health_model.pkl")

# Streamlit App
st.set_page_config(page_title="AI Mental Health Diagnosis", page_icon="🧠", layout="wide")

st.title("🧠 AI-Powered Mental Health Diagnosis System")
st.markdown("### 🏥 A simple tool to help assess mental health conditions")
st.write("Fill in the symptoms below to get a preliminary diagnosis. This is not a substitute for professional medical advice.")

# Sidebar for additional info
st.sidebar.header("About")
st.sidebar.info("This AI system predicts potential mental health conditions based on user inputs.This AI system harnesses the power of machine learning to predict potential mental health conditions based on user inputs.  this innovative tool is designed to provide accessible, user-friendly insights. By analyzing patterns and responses, the system offers a preliminary understanding that can guide users toward seeking professional assistance or further resources. It aims to support mental health awareness and empower individuals to take proactive steps in their well-being journey ")
st.sidebar.markdown("📌 **Disclaimer:** Always consult a healthcare professional for an accurate diagnosis.")

# User input fields
st.subheader("📝 Select Your Symptoms")
user_input = {}
cols = list(X.columns)
col1, col2 = st.columns(2)

for i, col in enumerate(cols):
    if i % 2 == 0:
        user_input[col] = col1.radio(f"{col.replace('.', ' ').title()}", ["No", "Yes"], index=0)
    else:
        user_input[col] = col2.radio(f"{col.replace('.', ' ').title()}", ["No", "Yes"], index=0)

# Convert input to numerical format
input_data = np.array([1 if user_input[col] == "Yes" else 0 for col in X.columns]).reshape(1, -1)

if st.button("🔍 Diagnose", use_container_width=True):
    model, le = joblib.load("mental_health_model.pkl")
    prediction = model.predict(input_data)
    diagnosis = le.inverse_transform(prediction)[0]

    st.success(f"### 🏥 Predicted Mental Health Condition: **{diagnosis}**")
    st.markdown("**📢 Note:** This is just a preliminary assessment. Seek professional help for further evaluation.")




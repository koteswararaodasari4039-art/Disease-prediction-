import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("disease_model.pkl")

st.title("🩺 Disease Prediction App")
st.write("Select symptoms and predict possible disease")

# Symptoms list (same order as dataset)
symptoms = [
    "itching", "skin_rash", "fever", "cough", "fatigue",
    "headache", "nausea", "vomiting", "diarrhea", "joint_pain",
    "chest_pain", "abdominal_pain", "breathlessness",
    "sore_throat", "runny_nose"
]

user_input = []

for symptom in symptoms:
    value = st.checkbox(symptom)
    user_input.append(1 if value else 0)

if st.button("Predict"):
    prediction = model.predict([user_input])
    st.success(f"Predicted Disease: {prediction[0]}")
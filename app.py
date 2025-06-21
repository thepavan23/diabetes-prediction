import streamlit as st
import numpy as np
import joblib

# Load scaler
scaler = joblib.load('scaler.pkl')

# Load models
models = {
    "Support Vector Machine": joblib.load("svm_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl"),
    "K-Nearest Neighbors": joblib.load("knn_model.pkl")
}

# Title
st.title("🩺 Diabetes Prediction Web App")
st.markdown("Predict diabetes using different ML models (SVM, Decision Tree, KNN).")

# Select model
model_choice = st.selectbox("Select a model for prediction:", list(models.keys()))

# User input
st.header("Enter Patient Details")
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0.0, 200.0, 120.0)
bp = st.number_input("Blood Pressure", 0.0, 150.0, 70.0)
skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
insulin = st.number_input("Insulin Level", 0.0, 900.0, 80.0)
bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 33)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    
    selected_model = models[model_choice]
    prediction = selected_model.predict(scaled_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    
    st.success(f"Prediction using **{model_choice}**: {result}")

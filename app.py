import streamlit as st
import numpy as np
import pickle

# Set page config (should be at the very top ideally)
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# Load model
with open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\diabetes prediction\model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\diabetes prediction\scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Optional: Load and display image
st.image(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\diabetes prediction\img.jpg", use_container_width=True)

# Title and instructions
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the patient's medical details in the sidebar to check the diabetes prediction.")

# Sidebar input fields
st.sidebar.header("📝 Patient Medical Data")
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, step=1)
glucose = st.sidebar.number_input("Glucose Level", 0, 200)
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", 0, 140)
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", 0, 100)
insulin = st.sidebar.number_input("Insulin Level (mu U/ml)", 0, 900)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.sidebar.number_input("Age", 0, 120)

# Predict button
if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    if np.all(input_data == 0):
        st.warning("⚠️ Please enter valid input values in the sidebar.")
    else:
        # Scale input and make prediction
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction]
        if prediction == 1:
            st.error("🧪 **The model predicts: Diabetes Positive**")
        else:
            st.success("✅ **The model predicts: Diabetes Negative**")

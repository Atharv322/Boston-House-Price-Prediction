import streamlit as st 
import numpy as np
import pickle

# load the model
with open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\boston house price\model.pkl", "rb") as f:
    model = pickle.load(f)
    
# App title
st.title("🏠 House Price Prediction")

features = ["CRIM (Per capita crime rate)", "ZN (Proportion of residential land zoned for lots over 25,000 sq.ft.)", "INDUS (Proportion of non-retail business acres)", "CHAS (Charles River dummy variable -0 or 1)", "NOX (Nitric oxide concentration)", "RM (Average number of rooms per dwelling)", "AGE (Proportion of owner-occupied units built before 1940)","DIS (Weighted distances to employment centers)", "RAD (Accessibility to radial highways)", "TAX (Property tax rate)", "PTRATIO (Pupil-teacher ratio)", "B (1000(Bk - 0.63)^2 where Bk is proportion of Black residents)", "LSTAT (% lower status of the population)"]

st.write("Enter House Features...")

input_data = []
for feature in features:
    value = st.number_input(f"{feature}: ", format="%.5f")
    input_data.append(value)
    
# Prediction
if st.button("Predict Price"):
    if all(v == 0.0 for v in input_data):
        st.warning("⚠️ Please enter values before predicting.")
    else:
        input_array = np.array(input_data).reshape(1,-1)
        prediction = model.predict(input_array)
        st.success(f"💰 Estimated Price: ${prediction[0]:.2f}")
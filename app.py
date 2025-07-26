import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Set page configuration
st.set_page_config(page_title="🏠 Boston Housing Price Prediction", layout="centered")

# App Title
st.title("🏠 Boston Housing Price Prediction")
st.markdown("Enter the details below to predict the housing price.")

# Sidebar for inputs
st.sidebar.header("Input Features")

def user_input_features():
    CRIM = st.sidebar.slider('CRIM (Per capita crime rate)', 0.0, 100.0, 0.5)
    ZN = st.sidebar.slider('ZN (Residential land zoned)', 0.0, 100.0, 25.0)
    INDUS = st.sidebar.slider('INDUS (Non-retail business acres)', 0.0, 30.0, 5.0)
    CHAS = st.sidebar.selectbox('CHAS (Charles River dummy variable)', (0, 1))
    NOX = st.sidebar.slider('NOX (Nitric oxides concentration)', 0.0, 1.0, 0.5)
    RM = st.sidebar.slider('RM (Average number of rooms)', 3.0, 9.0, 6.0)
    AGE = st.sidebar.slider('AGE (Proportion of owner units built before 1940)', 0.0, 100.0, 50.0)
    DIS = st.sidebar.slider('DIS (Distance to employment centres)', 1.0, 12.0, 5.0)
    RAD = st.sidebar.slider('RAD (Accessibility to highways)', 1, 24, 5)
    TAX = st.sidebar.slider('TAX (Property tax rate)', 100, 800, 300)
    PTRATIO = st.sidebar.slider('PTRATIO (Pupil-teacher ratio)', 10.0, 30.0, 18.0)
    B = st.sidebar.slider('B (Proportion of black population)', 0.0, 400.0, 350.0)
    LSTAT = st.sidebar.slider('LSTAT (% lower status population)', 1.0, 40.0, 12.0)

    features = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
    return np.array([features])

# Get user input
input_data = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"🏡 Predicted House Price: ${prediction[0]*1000:.2f}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

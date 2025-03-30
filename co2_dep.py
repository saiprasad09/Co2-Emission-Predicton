import pandas as pd
import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
model = pickle.load(open('co2_rf_model .pkl','rb'))

st.title('CO₂ Prediction Model Deployment using Random Forest Regression')

feature_names = [
    "engine_size", "cylinders", "fuel_consumption_city",
    "fuel_consumption_hwy", "fuel_consumption_comb(l/100km)"
]

user_input = []
all_zeros = True  # Flag to check if all inputs are zero

for feature in feature_names:  # Numerical inputs
    value = st.sidebar.number_input(f"{feature}", value=0.0)
    user_input.append(value)
    if value != 0:  # If any value is non-zero, update the flag
        all_zeros = False

input_array = np.array(user_input).reshape(1, -1)

st.subheader("User Input Features")
st.write(pd.DataFrame([user_input], columns=feature_names))

if all_zeros:
    st.error("Please enter valid values. All inputs cannot be zero.")
else:
    prediction = model.predict(input_array)
    st.subheader("Predicted CO₂ Emissions")
    st.write(f"**{prediction[0]:.2f} g/km**")

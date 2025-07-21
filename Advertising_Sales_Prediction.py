import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("ridge.pkl", "rb"))

# App Title
st.title("📢 Advertising Sales Prediction App")
st.markdown("Predict product sales based on advertising spend using a trained Linear Regression model.")

# Input widgets
tv = st.slider("TV Advertising Budget (in $1000s)", 0.0, 300.0, 100.0)
radio = st.slider("Radio Advertising Budget (in $1000s)", 0.0, 50.0, 25.0)
newspaper = st.slider("Newspaper Advertising Budget (in $1000s)", 0.0, 120.0, 30.0)

# Prediction
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Sales: **{prediction:.2f}** units")

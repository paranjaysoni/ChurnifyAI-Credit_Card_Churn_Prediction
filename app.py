import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import tensorflow as tf
import time
from io import BytesIO

@st.cache_resource
def load_pickle_from_url(url):
    response = requests.get(url)
    return pickle.load(BytesIO(response.content))

@st.cache_resource
def load_model_from_url(url):
    response = requests.get(url)
    with open("temp_model.h5", "wb") as f:
        f.write(response.content)
    return tf.keras.models.load_model("temp_model.h5")

url_geo = "https://huggingface.co/soniparanjay/ChurnifyAI/resolve/main/onehot_encoder_geo.pkl"
url_gender = "https://huggingface.co/soniparanjay/ChurnifyAI/resolve/main/label_encoder_gender.pkl"
url_scaler = "https://huggingface.co/soniparanjay/ChurnifyAI/resolve/main/scaler.pkl"
url_model = "https://huggingface.co/soniparanjay/ChurnifyAI/resolve/main/model.h5"

onehot_encoder_geo = load_pickle_from_url(url_geo)
label_encoder_gender = load_pickle_from_url(url_gender)
scaler = load_pickle_from_url(url_scaler)
model = load_model_from_url(url_model)

st.set_page_config(page_title="Churnify AI", page_icon="💳", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        💳 Churnify AI <br> <span style='font-size:20px; color:gray;'>Credit Card Churn Prediction</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.write("✨ Enter customer details below to predict whether they will churn or stay loyal.")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("⚧ Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92, 30)
    tenure = st.slider("📅 Tenure (Years)", 0, 10, 3)
    num_of_products = st.slider("🛒 Number of Products", 1, 4, 1)

with col2:
    credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650)
    balance = st.number_input("🏦 Balance", min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("💰 Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)
    has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
    is_active_member = st.selectbox("🔥 Is Active Member", [0, 1])

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

geo_encoded = onehot_encoder_geo.transform([[geography]])
if not isinstance(geo_encoded, np.ndarray):
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

if st.button("🚀 Predict Churn"):
    with st.spinner("⏳ Analyzing customer profile..."):
        time.sleep(2)  
        prediction = model.predict(input_data_scaled)
        prob = prediction[0][0]

    if prob > 0.5:
        st.markdown(
            "<h2 style='text-align:center; color: red;'>❌ The customer is likely to Leave</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h2 style='text-align:center; color: green;'>✅ The customer is likely to STAY</h2>",
            unsafe_allow_html=True,
        )

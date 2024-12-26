import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

st.markdown("""
<style>
    /* Main title styling */
    h1 {  /* Streamlit title class */
        font-size: 40px;
        color: #db2777;
        font-weight: bold;
        margin-bottom: 30px;
        text-transform: uppercase;
        background: linear-gradient(to right, #db2777, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
</style>
""", unsafe_allow_html=True)

## Load the model
model = tf.keras.models.load_model('model.h5')

## Load the pickle files
with open("gender_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)

with open("one_hot_encoder_geo.pkl", "rb") as file:
    one_hot_encoder_geo = pickle.load(file)

with open("scalar.pkl", "rb") as file:
    scalar = pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")
st.write("This is a simple app to predict whether a customer will leave the bank or not")

## User input
geography = st.selectbox("Select Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender", gender_encoder.classes_)
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
num_products = st.selectbox("Number of Products", [1,2,3,4])
has_cr_card = st.selectbox("Has Credit Card (1 for YES & 0 for NO)", [0, 1])
is_active_member = st.selectbox("Is Active Member (1 for YES & 0 for NO)", [0, 1])
estimated_salary = st.number_input("Estimated Salary")

## Input for prediction
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoder.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

## One hot encoding Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = ["France", "Germany", "Spain"])
input_data = pd.concat([pd.DataFrame(input_data), geo_encoded_df], axis=1)

## Scaling the data
input_data_scaled = scalar.transform(input_data)

## Prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write("Customer will leave the bank")
    st.write(f"Probability of customer leaving the bank: {int(prediction_prob*100)}%")
else:
    st.write("Customer will stay with the bank")
    st.write(f"Probability of customer staying with the bank: {int((1 - prediction_prob)*100)}%")
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=492, random_state=42)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)
    X = balanced_data.drop(columns="Class", axis=1)
    y = balanced_data["Class"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = train_model()

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details below to check for fraud.")

# Define input fields based on dataset (removing 'Class')
input_data = {}
feature_names = pd.read_csv("creditcard.csv", nrows=1).drop(columns="Class").columns
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.01)

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("⚠️ Transaction is Fraudulent!")
    else:
        st.success("✅ Transaction is Legitimate.")

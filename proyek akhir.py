import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model pipeline
model = joblib.load("bank_marketing_pipeline_xgb.pkl")

st.set_page_config(page_title="Bank Deposit Prediction", page_icon="🏦")
st.title("🏦 Prediksi Deposit Bank")
st.markdown("Masukkan data nasabah untuk memprediksi apakah akan membuka deposito.")

# --- Input Form ---
st.sidebar.header("Input Data Nasabah")

age = st.sidebar.slider("Age", 18, 95, 35)
balance = st.sidebar.number_input("Balance", -10000, 100000, 1000)
day = st.sidebar.slider("Day (of month)", 1, 31, 15)
duration = st.sidebar.number_input("Duration (detik)", 0, 5000, 200)
campaign = st.sidebar.slider("Campaign (jumlah kontak)", 1, 50, 2)
pdays = st.sidebar.number_input("Pdays (-1 = belum pernah)", -1, 999, -1)
previous = st.sidebar.slider("Previous", 0, 50, 0)

job = st.sidebar.selectbox("Job", ['admin.','blue-collar','entrepreneur','housemaid',
                                    'management','retired','self-employed','services',
                                    'student','technician','unemployed','unknown'])
marital = st.sidebar.selectbox("Marital", ['divorced','married','single'])
education = st.sidebar.selectbox("Education", ['primary','secondary','tertiary','unknown'])
default = st.sidebar.selectbox("Default", ['no','yes'])
contact = st.sidebar.selectbox("Contact", ['cellular','telephone','unknown'])
month = st.sidebar.selectbox("Month", ['jan','feb','mar','apr','may','jun',
                                        'jul','aug','sep','oct','nov','dec'])
poutcome = st.sidebar.selectbox("Poutcome", ['failure','other','success','unknown'])
deposit = st.sidebar.selectbox("Deposit (target sebelumnya)", ['no','yes'])

# --- Predict ---
if st.button("🔍 Prediksi"):
    input_data = pd.DataFrame([{
        'age': age, 'balance': balance, 'day': day,
        'duration': duration, 'campaign': campaign,
        'pdays': pdays, 'previous': previous,
        'job': job, 'marital': marital, 'education': education,
        'default': default, 'contact': contact, 'month': month,
        'poutcome': poutcome, 'deposit': deposit
    }])

    proba = model.predict_proba(input_data)[0][1]
    pred = "✅ YA — Kemungkinan membuka deposito" if proba >= 0.4 else "❌ TIDAK — Kemungkinan tidak membuka deposito"

    st.subheader("Hasil Prediksi:")
    st.success(pred) if proba >= 0.4 else st.error(pred)
    st.metric("Probabilitas Deposito", f"{proba:.2%}")

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# ── Load model ───────────────────────────────────────────────
model = XGBClassifier()
model.load_model("xgb_model.json")

# ── Fungsi buat input vector 51 fitur ───────────────────────
def build_input(age, balance, day, duration, campaign, pdays, previous,
                job, marital, education, default_, housing, loan,
                contact, month, poutcome):

    # Semua kolom hasil OHE (urutan harus sama persis dengan training)
    cols = [
        'num__age','num__balance','num__day','num__duration','num__campaign','num__pdays','num__previous',
        'cat__job_admin.','cat__job_blue-collar','cat__job_entrepreneur','cat__job_housemaid',
        'cat__job_management','cat__job_retired','cat__job_self-employed','cat__job_services',
        'cat__job_student','cat__job_technician','cat__job_unemployed','cat__job_unknown',
        'cat__marital_divorced','cat__marital_married','cat__marital_single',
        'cat__education_primary','cat__education_secondary','cat__education_tertiary','cat__education_unknown',
        'cat__default_no','cat__default_yes',
        'cat__housing_no','cat__housing_yes',
        'cat__loan_no','cat__loan_yes',
        'cat__contact_cellular','cat__contact_telephone','cat__contact_unknown',
        'cat__month_apr','cat__month_aug','cat__month_dec','cat__month_feb','cat__month_jan',
        'cat__month_jul','cat__month_jun','cat__month_mar','cat__month_may','cat__month_nov',
        'cat__month_oct','cat__month_sep',
        'cat__poutcome_failure','cat__poutcome_other','cat__poutcome_success','cat__poutcome_unknown'
    ]

    row = {c: 0.0 for c in cols}

    # Numerik (tanpa scaling — XGBoost tidak butuh StandardScaler)
    row['num__age']      = age
    row['num__balance']  = balance
    row['num__day']      = day
    row['num__duration'] = duration
    row['num__campaign'] = campaign
    row['num__pdays']    = pdays
    row['num__previous'] = previous

    # OHE kategorik
    row[f'cat__job_{job}']           = 1.0
    row[f'cat__marital_{marital}']   = 1.0
    row[f'cat__education_{education}']= 1.0
    row[f'cat__default_{default_}']  = 1.0
    row[f'cat__housing_{housing}']   = 1.0
    row[f'cat__loan_{loan}']         = 1.0
    row[f'cat__contact_{contact}']   = 1.0
    row[f'cat__month_{month}']       = 1.0
    row[f'cat__poutcome_{poutcome}'] = 1.0

    return pd.DataFrame([row])[cols]

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Bank Deposit Prediction", page_icon="🏦")
st.title("🏦 Prediksi Deposit Bank")
st.markdown("Masukkan data nasabah untuk memprediksi apakah akan membuka deposito.")

st.sidebar.header("Input Data Nasabah")

age      = st.sidebar.slider("Age", 18, 95, 35)
balance  = st.sidebar.number_input("Balance", -10000, 100000, 1000)
day      = st.sidebar.slider("Day (of month)", 1, 31, 15)
duration = st.sidebar.number_input("Duration (detik)", 0, 5000, 200)
campaign = st.sidebar.slider("Campaign (jumlah kontak)", 1, 50, 2)
pdays    = st.sidebar.number_input("Pdays (-1 = belum pernah)", -1, 999, -1)
previous = st.sidebar.slider("Previous", 0, 50, 0)

job       = st.sidebar.selectbox("Job", ['admin.','blue-collar','entrepreneur','housemaid',
                                          'management','retired','self-employed','services',
                                          'student','technician','unemployed','unknown'])
marital   = st.sidebar.selectbox("Marital", ['divorced','married','single'])
education = st.sidebar.selectbox("Education", ['primary','secondary','tertiary','unknown'])
default_  = st.sidebar.selectbox("Default", ['no','yes'])
housing   = st.sidebar.selectbox("Housing Loan", ['no','yes'])
loan      = st.sidebar.selectbox("Personal Loan", ['no','yes'])
contact   = st.sidebar.selectbox("Contact", ['cellular','telephone','unknown'])
month     = st.sidebar.selectbox("Month", ['jan','feb','mar','apr','may','jun',
                                            'jul','aug','sep','oct','nov','dec'])
poutcome  = st.sidebar.selectbox("Poutcome", ['failure','other','success','unknown'])

if st.button("🔍 Prediksi"):
    X = build_input(age, balance, day, duration, campaign, pdays, previous,
                    job, marital, education, default_, housing, loan,
                    contact, month, poutcome)

    proba = model.predict_proba(X)[0][1]
    pred  = "✅ YA — Kemungkinan membuka deposito" if proba >= 0.4 else "❌ TIDAK — Kemungkinan tidak membuka deposito"

    st.subheader("Hasil Prediksi:")
    st.success(pred) if proba >= 0.4 else st.error(pred)
    st.metric("Probabilitas Deposito", f"{proba:.2%}")

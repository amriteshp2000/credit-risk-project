# components/form.py
import streamlit as st
import pandas as pd

def get_user_input_form():
    st.subheader("📋 Enter Applicant Information")

    dob_days = st.number_input("🧓 Date of Birth (in days before today)", min_value=0, max_value=30000, value=12000)
    income = st.number_input("💰 Total Annual Income (₹)", min_value=0, value=200000)
    credit = st.number_input("🏦 Loan Amount Requested (₹)", min_value=0, value=100000)
    annuity = st.number_input("📆 Monthly Installment Amount (₹)", min_value=0, value=5000)
    goods_price = st.number_input("🛍️ Value of Goods for Loan (₹)", min_value=0, value=90000)
    employment_years = st.slider("👨‍💼 Years of Employment", 0, 50, 5)
    days_credit = st.number_input("📜 Days since first credit record", min_value=0, max_value=10000, value=1000)

    education = st.selectbox("🎓 Highest Education Attained", ["Secondary", "Higher"])
    marital_status = st.selectbox("💍 Marital Status", ["Married", "Other"])
    own_car = st.radio("🚗 Do you own a car?", ["Yes", "No"])

    # One-hot encode features manually
    education_secondary = 1 if education == "Secondary" else 0
    education_higher = 1 if education == "Higher" else 0
    is_married = 1 if marital_status == "Married" else 0
    flag_own_car = 1 if own_car == "Yes" else 0

    input_data = pd.DataFrame([{
        "DAYS_BIRTH": dob_days,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "EMPLOYMENT_YEARS": employment_years,
        "DAYS_CREDIT": days_credit,
        "NAME_EDUCATION_TYPE_Secondary / secondary special": education_secondary,
        "NAME_EDUCATION_TYPE_Higher education": education_higher,
        "NAME_FAMILY_STATUS_Married": is_married,
        "FLAG_OWN_CAR": flag_own_car
    }])

    return input_data

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Bank Churn Forensic Tool", layout="wide")

@st.cache_resource # Keeps the model in memory for speed
def load_assets():
    # Update paths if you put them in a /models folder
    model = joblib.load('forensic_churn_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- 2. UI HEADER ---
st.title("🏦 Bank Churn Forensic Diagnostic Tool")
st.markdown("""
This tool uses a **High-Recall Logistic Regression model (74%)** to identify at-risk customers 
based on forensic behavioral patterns.
""")

# --- 3. INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    age = st.slider("Age", 18, 92, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    geo = st.selectbox("Geography", ["France", "Germany", "Spain"])

with col2:
    st.header("💰 Financials")
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step = 1000.0)
    salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0, step = 1000.0)
    credit_score = st.slider("Credit Score", 350, 850, 650)

with col3:
    st.header("📊 Engagement")
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    is_active = st.radio("Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    has_card = st.radio("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# --- 4. FORENSIC FEATURE ENGINEERING ---
# The app must replicate your training logic exactly
balance_salary_ratio = balance / (salary + 1)
tenure_by_age = tenure / age
cs_per_product = credit_score / products
is_inactive_whale = 1 if (balance > 100000 and is_active == 0) else 0

# Mapping Categorical to match 'drop_first=True'
geo_germany = 1 if geo == "Germany" else 0
geo_spain = 1 if geo == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# --- 5. PREDICTION PIPELINE ---
# Create the input array in the EXACT order of your X_train columns

feature_columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
    'IsActiveMember', 'EstimatedSalary','Geography_Germany', 'Geography_Spain', 
    'Gender_Male', 'Balance_Salary_Ratio', 'Tenure_By_Age', 'Is_Inactive_Whale', 
    'Credit_Score_Per_Product'
]

input_df = pd.DataFrame([[
    credit_score, age, tenure, balance, products, has_card, is_active, salary,
    geo_germany, geo_spain, gender_male, 
    balance_salary_ratio, tenure_by_age, is_inactive_whale, cs_per_product
]], columns=feature_columns)


# Scale the data using the saved scaler
scaled_input = scaler.transform(input_df)
prediction_prob = model.predict_proba(scaled_input)[0][1]

# --- 6. RESULTS & INTERVENTION ---
st.divider()
st.header("🔍 Diagnostic Verdict")

if prediction_prob > 0.5:
    st.error(f"### ⚠️ HIGH RISK: {prediction_prob:.2%} Probability of Churn")
    
    # Custom Logic for Intervention based on features
    if cs_per_product > 400 and products == 1:
        st.warning("**Priority Intervention:** Target with the 'Anchor Offer' (Premium Bundle).")
    elif is_inactive_whale:
        st.warning("**Priority Intervention:** Trigger 'Concierge Re-engagement' call.")
else:
    st.success(f"### ✅ STABLE: {prediction_prob:.2%} Probability of Churn")

# Show the full strategy table for the user
with st.expander("View Full Strategic Intervention Table"):
    st.table({
        "Segment": ["Underutilized Elite", "Inactive Whales", "German Leak", "Vulnerable Seniors"],
        "Priority": ["CRITICAL", "HIGH", "MEDIUM", "MEDIUM"],
        "Action": ["Anchor Offer", "Concierge Call", "Regional Audit", "Legacy Program"]
    })


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


MODEL_FILE = "fraud_detection_model.pkl"

# Check if the model file exists
if not os.path.exists(MODEL_FILE):
    st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure the model from the training step is saved in the current directory.")
    st.stop()

# Load the trained model pipeline
@st.cache_resource
def load_model(filename):
    """Loads the pre-trained model and preprocessing pipeline."""
    try:
        model = joblib.load(filename)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model_pipeline = load_model(MODEL_FILE)

# Define options for categorical features (pulled from previous analysis)
CAT_OPTIONS = {
    'PatientGender': ['M', 'F'],
    'ProviderSpecialty': ['Cardiology', 'Pediatrics', 'Neurology', 'General Practice', 'Orthopedics'],
    'ClaimStatus': ['Pending', 'Approved', 'Denied'],
    'PatientMaritalStatus': ['Married', 'Single', 'Divorced', 'Widowed'],
    'PatientEmploymentStatus': ['Retired', 'Student', 'Employed', 'Unemployed'],
    'ClaimType': ['Routine', 'Emergency', 'Inpatient', 'Outpatient'],
    'ClaimSubmissionMethod': ['Paper', 'Online', 'Phone']
}

# Define the order of columns used during training
FEATURE_COLUMNS = [
    "PatientAge", "ClaimAmount", "PatientIncome", "PatientGender", 
    "ProviderSpecialty", "ClaimStatus", "PatientMaritalStatus", 
    "PatientEmploymentStatus", "ClaimType", "ClaimSubmissionMethod"
]

# --- Streamlit UI ---

st.set_page_config(page_title="Health Insurance Fraud Detector", layout="wide")
st.title("🏥 Health Insurance Claim Fraud Detector")
st.markdown("Enter the claim details below to predict the likelihood of fraud.")

# Use columns for a better layout
col1, col2, col3 = st.columns(3)

# --- Feature Input Fields ---

# Numeric Inputs (Col 1)
with col1:
    st.header("Patient & Claim Details")
    patient_age = st.slider("Patient Age", 0, 100, 45)
    claim_amount = st.number_input("Claim Amount ($)", min_value=100.00, max_value=100000.00, value=5000.00, step=100.00)
    patient_income = st.number_input("Patient Income ($)", min_value=0.00, max_value=500000.00, value=75000.00, step=500.00)

# Categorical Inputs (Col 2 & Col 3)
with col2:
    st.header("Claim Characteristics")
    provider_specialty = st.selectbox("Provider Specialty", CAT_OPTIONS['ProviderSpecialty'])
    claim_status = st.selectbox("Claim Status", CAT_OPTIONS['ClaimStatus'])
    claim_type = st.selectbox("Claim Type", CAT_OPTIONS['ClaimType'])
    claim_submission = st.selectbox("Submission Method", CAT_OPTIONS['ClaimSubmissionMethod'])

with col3:
    st.header("Patient Demographics")
    patient_gender = st.selectbox("Patient Gender", CAT_OPTIONS['PatientGender'])
    marital_status = st.selectbox("Marital Status", CAT_OPTIONS['PatientMaritalStatus'])
    employment_status = st.selectbox("Employment Status", CAT_OPTIONS['PatientEmploymentStatus'])


st.markdown("---") # Separator

# Prediction Button
if st.button("Predict Fraud Risk", type="primary"):
    
    # 1. Gather all inputs into a dictionary
    input_data = {
        "PatientAge": patient_age,
        "ClaimAmount": claim_amount,
        "PatientIncome": patient_income,
        "PatientGender": patient_gender,
        "ProviderSpecialty": provider_specialty,
        "ClaimStatus": claim_status,
        "PatientMaritalStatus": marital_status,
        "PatientEmploymentStatus": employment_status,
        "ClaimType": claim_type,
        "ClaimSubmissionMethod": claim_submission
    }
    
    # 2. Convert to a DataFrame, ensuring correct column order and single row
    new_claim = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    
    # 3. Make Prediction
    try:
        # Prediction (0 or 1)
        prediction = model_pipeline.predict(new_claim)[0]
        
        # Prediction Probability (for risk assessment)
        probability = model_pipeline.predict_proba(new_claim)[0, 1] # Probability of class 1 (Fraud)
        
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"🚨 HIGH RISK: This claim is predicted as **FRAUDULENT**.")
        else:
            st.success("✅ LOW RISK: This claim is predicted as **NOT FRAUDULENT**.")
        
        st.metric("Fraud Probability (P=1)", f"{probability:.2%}", delta=None)
        
        st.info("The model pipeline handles all necessary scaling and one-hot encoding internally.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

# App title
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Predict which customers are likely to leave • **79.5% Accuracy**")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_churn_model_lightgbm.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_model()

# Customer input form
st.header("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 350, 850, 650)
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0, 1000.0)

with col2:
    num_products = st.slider("Number of Products", 1, 4, 2)
    estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0, 1000.0)
    geography = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.radio("Gender", ["Male", "Female"])

col3, col4 = st.columns(2)
with col3:
    has_credit_card = st.radio("Has Credit Card?", ["Yes", "No"])
with col4:
    is_active_member = st.radio("Is Active Member?", ["Yes", "No"])

# Prediction function
def predict_churn():
    # Prepare input
    input_data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_credit_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active_member == "Yes" else 0,
        'EstimatedSalary': estimated_salary,
        'Gender_encoded': 1 if gender == "Male" else 0,
        'Geo_France': 1 if geography == "France" else 0,
        'Geo_Germany': 1 if geography == "Germany" else 0,
        'Geo_Spain': 1 if geography == "Spain" else 0
    }
    
    df = pd.DataFrame([input_data])
    
    # Scale numerical features
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Make prediction
    churn_probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]
    
    return churn_probability, prediction

# Prediction button
st.markdown("---")
if st.button("🎯 Predict Churn Probability", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please check deployment files.")
    else:
        churn_prob, prediction = predict_churn()
        
        # Display results
        st.header("📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        
        with col2:
            status = "🚨 WILL CHURN" if prediction == 1 else "✅ WILL STAY"
            st.metric("Prediction", status)
        
        # Risk assessment
        if churn_prob > 0.7:
            st.error("🔴 HIGH RISK: Immediate action needed!")
        elif churn_prob > 0.3:
            st.warning("🟡 MEDIUM RISK: Monitor closely")
        else:
            st.success("🟢 LOW RISK: Customer is stable")
        
        # Recommendations
        if churn_prob > 0.5:
            st.info("💡 **Retention Suggestions:**")
            suggestions = []
            
            if geography == "Germany":
                suggestions.append("• Germany-specific retention program")
            if num_products >= 3:
                suggestions.append("• Review product bundle")
            if is_active_member == "No":
                suggestions.append("• Customer activation campaign")
            
            for suggestion in suggestions:
                st.write(suggestion)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Powered by LightGBM • 79.5% Churn Detection</p>
</div>
""", unsafe_allow_html=True)

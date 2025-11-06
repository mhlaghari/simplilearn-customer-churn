import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Bank Churn Predictor - Fixed",
    page_icon="🏦",
    layout="centered"
)

# App title
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("**Final Fixed Version** • 79.5% Churn Detection Accuracy")

# Model loading with exact feature order
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_churn_model_lightgbm.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, False

model, scaler, model_loaded = load_model()

# Define the EXACT feature order the model expects
FEATURE_ORDER = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender_encoded',
    'Geo_France', 'Geo_Germany', 'Geo_Spain'
]

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

has_credit_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
is_active_member = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)

# Enhanced prediction with exact feature matching
def predict_churn_fixed():
    # Prepare input with EXACT feature order and data types
    input_data = {
        'CreditScore': float(credit_score),
        'Age': float(age),
        'Tenure': float(tenure),
        'Balance': float(balance),
        'NumOfProducts': float(num_products),
        'HasCrCard': 1 if has_credit_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active_member == "Yes" else 0,
        'EstimatedSalary': float(estimated_salary),
        'Gender_encoded': 1 if gender == "Male" else 0,
        'Geo_France': 1 if geography == "France" else 0,
        'Geo_Germany': 1 if geography == "Germany" else 0,
        'Geo_Spain': 1 if geography == "Spain" else 0
    }
    
    # Create DataFrame with EXACT feature order
    df = pd.DataFrame([input_data])[FEATURE_ORDER]
    
    # Scale ONLY the numerical features that were scaled during training
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Make prediction
    probabilities = model.predict_proba(df_scaled)[0]
    prediction = model.predict(df_scaled)[0]
    
    return probabilities[1], prediction, df_scaled

# Prediction section
st.markdown("---")
st.header("🎯 Prediction Results")

if st.button("🔍 Predict Churn Probability", type="primary", use_container_width=True):
    if not model_loaded:
        st.error("Model not loaded. Please check deployment files.")
    else:
        churn_prob, prediction, processed_data = predict_churn_fixed()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        
        with col2:
            status = "🚨 WILL CHURN" if prediction == 1 else "✅ WILL STAY"
            st.metric("Prediction", status)
            
        with col3:
            confidence = churn_prob if prediction == 1 else (1 - churn_prob)
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Risk assessment
        if churn_prob > 0.7:
            st.error("🔴 HIGH RISK: Immediate retention action needed!")
        elif churn_prob > 0.4:
            st.warning("🟡 MEDIUM RISK: Proactive monitoring recommended")
        else:
            st.success("🟢 LOW RISK: Customer is likely to stay")
        
        # Debug information (collapsible)
        with st.expander("🔧 Technical Details (For Debugging)"):
            st.write("**Processed Features:**")
            st.write(processed_data.iloc[0].to_dict())
            st.write(f"**Probabilities:** No Churn: {1-churn_prob:.3f}, Churn: {churn_prob:.3f}")

        # Smart recommendations
        if churn_prob > 0.3:
            st.info("💡 **Recommended Retention Actions:**")
            
            recommendations = []
            if geography == "Germany":
                recommendations.append("• **Germany-focused retention**: 32% higher churn rate in Germany")
            if gender == "Female":
                recommendations.append("• **Female engagement program**: 25% higher churn rate among females")
            if age > 40:
                recommendations.append("• **Senior customer program**: Older customers more likely to churn")
            if num_products >= 3:
                recommendations.append("• **Product bundle review**: 3+ products have 82%+ churn risk")
            if is_active_member == "No":
                recommendations.append("• **Activation campaign**: Inactive members 47% more likely to churn")
            if balance > 100000:
                recommendations.append("• **High-value retention**: Protect valuable high-balance customers")
            
            for rec in recommendations:
                st.write(rec)

# Add preset test cases in sidebar
st.sidebar.header("🧪 Test Cases")
st.sidebar.markdown("Try these scenarios:")

if st.sidebar.button("Low Risk Customer"):
    st.session_state.credit_score = 800
    st.session_state.age = 30
    st.session_state.tenure = 9
    st.session_state.balance = 50000
    st.session_state.num_products = 2
    st.session_state.estimated_salary = 100000
    st.session_state.geography = "France"
    st.session_state.gender = "Male"
    st.session_state.has_credit_card = "Yes"
    st.session_state.is_active_member = "Yes"
    st.rerun()

if st.sidebar.button("High Risk Customer"):
    st.session_state.credit_score = 450
    st.session_state.age = 55
    st.session_state.tenure = 1
    st.session_state.balance = 200000
    st.session_state.num_products = 4
    st.session_state.estimated_salary = 40000
    st.session_state.geography = "Germany"
    st.session_state.gender = "Female"
    st.session_state.has_credit_card = "No"
    st.session_state.is_active_member = "No"
    st.rerun()

# Initialize session state
if 'credit_score' not in st.session_state:
    st.session_state.credit_score = 650
    st.session_state.age = 40
    st.session_state.tenure = 5
    st.session_state.balance = 50000
    st.session_state.num_products = 2
    st.session_state.estimated_salary = 75000
    st.session_state.geography = "France"
    st.session_state.gender = "Male"
    st.session_state.has_credit_card = "Yes"
    st.session_state.is_active_member = "Yes"

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Final Fixed Version • Exact Feature Matching • 79.5% Accuracy</p>
</div>
""", unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Bank Churn Predictor Pro",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_churn_model_lightgbm.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize
model, scaler = load_artifacts()

# App title
st.markdown('<h1 class="main-header">🏦 Bank Customer Churn Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown("""
**Powered by Machine Learning • 79.5% Churn Detection Accuracy**
""")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts customer churn probability using our trained LightGBM model.
    
    **Model Performance:**
    - 79.5% Churn Detection Rate
    - 86.2% ROC-AUC Score
    - Trained on 10,000 customers
    
    **Top Churn Factors:**
    1. Age
    2. Account Balance  
    3. Number of Products
    4. Geography
    5. Activity Status
    """)
    
    st.header("📊 Quick Stats")
    if model is not None:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Loaded")

# Main content - Customer input form
st.header("📝 Customer Information")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["🧮 Form Input", "📁 Batch Upload"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Demographic Info")
        geography = st.selectbox("Country", ["France", "Germany", "Spain"], 
                               help="Customer's country of residence")
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (Years)", 0, 10, 5,
                         help="Years as customer")
        
    with col2:
        st.subheader("💰 Financial Info")
        credit_score = st.slider("Credit Score", 350, 850, 650)
        balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0, 1000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0, 1000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
        
    col3, col4 = st.columns(2)
    with col3:
        has_credit_card = st.radio("Has Credit Card?", ["Yes", "No"])
    with col4:
        is_active_member = st.radio("Is Active Member?", ["Yes", "No"])

with tab2:
    st.subheader("📁 Upload Customer Data")
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=['csv'])
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())
            st.success(f"✅ Successfully loaded {len(batch_data)} customers")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Prediction function
def predict_churn_single(input_dict):
    """Predict churn for a single customer"""
    # Create input dataframe
    input_data = {
        'CreditScore': input_dict['credit_score'],
        'Age': input_dict['age'],
        'Tenure': input_dict['tenure'],
        'Balance': input_dict['balance'],
        'NumOfProducts': input_dict['num_products'],
        'HasCrCard': 1 if input_dict['has_credit_card'] == "Yes" else 0,
        'IsActiveMember': 1 if input_dict['is_active_member'] == "Yes" else 0,
        'EstimatedSalary': input_dict['estimated_salary'],
        'Gender_encoded': 1 if input_dict['gender'] == "Male" else 0,
        'Geo_France': 1 if input_dict['geography'] == "France" else 0,
        'Geo_Germany': 1 if input_dict['geography'] == "Germany" else 0,
        'Geo_Spain': 1 if input_dict['geography'] == "Spain" else 0
    }
    
    df = pd.DataFrame([input_data])
    
    # Scale numerical features
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Make prediction
    churn_probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]
    
    return churn_probability, prediction

# Prediction section
st.markdown("---")
st.header("🎯 Churn Prediction")

if st.button("🚀 Predict Churn Probability", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please check if model files are available.")
    else:
        # Prepare input
        input_dict = {
            'credit_score': credit_score,
            'geography': geography,
            'gender': gender,
            'age': age,
            'tenure': tenure,
            'balance': balance,
            'num_products': num_products,
            'has_credit_card': has_credit_card,
            'is_active_member': is_active_member,
            'estimated_salary': estimated_salary
        }
        
        churn_probability, prediction = predict_churn_single(input_dict)
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability:.1%}",
                delta="High Risk" if churn_probability > 0.5 else "Low Risk"
            )
        
        with col2:
            status = "🚨 WILL CHURN" if prediction == 1 else "✅ WILL STAY"
            st.metric(label="Prediction", value=status)
        
        with col3:
            confidence = churn_probability if prediction == 1 else (1 - churn_probability)
            st.metric(label="Confidence", value=f"{confidence:.1%}")
        
        # Risk assessment with colored box
        risk_class = "high-risk" if churn_probability > 0.7 else "medium-risk" if churn_probability > 0.3 else "low-risk"
        risk_message = {
            "high-risk": "🔴 HIGH RISK: Immediate retention action recommended!",
            "medium-risk": "🟡 MEDIUM RISK: Monitor and consider proactive engagement",
            "low-risk": "🟢 LOW RISK: Customer is likely to stay"
        }
        
        st.markdown(f'<div class="prediction-box {risk_class}"><h3>📊 Risk Assessment</h3><p>{risk_message[risk_class]}</p></div>', unsafe_allow_html=True)
        
        # Retention recommendations
        if churn_probability > 0.3:
            st.subheader("🛡️ Retention Action Plan")
            
            recommendations = []
            if geography == "Germany":
                recommendations.append("🇩🇪 **Germany-specific retention program** (32% higher churn rate)")
            if gender == "Female":
                recommendations.append("👩 **Female-focused engagement** (25% higher churn rate)")
            if age > 40:
                recommendations.append("🎂 **Senior customer program** (older customers more likely to churn)")
            if num_products >= 3:
                recommendations.append("📦 **Product bundle review** (3+ products have 82%+ churn rate)")
            if is_active_member == "No":
                recommendations.append("🔄 **Activation campaign** (inactive members 47% more likely to churn)")
            if balance > 100000:
                recommendations.append("💰 **High-value customer retention** (high balance customers more likely to leave)")
                
            for rec in recommendations:
                st.write(f"• {rec}")
        
        # Feature importance visualization
        st.subheader("🔍 Key Factors Influencing Prediction")
        
        # Feature importance data (from our model analysis)
        feature_importance = {
            'Age': 0.245,
            'Balance': 0.143,
            'Estimated Salary': 0.140,
            'Credit Score': 0.138,
            'Number of Products': 0.126,
            'Tenure': 0.081,
            'Active Member': 0.036,
            'Geography (Germany)': 0.025,
            'Gender': 0.022,
            'Credit Card': 0.019
        }
        
        # Create interactive chart
        fig = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title='Feature Importance in Churn Prediction',
            labels={'x': 'Importance', 'y': 'Features'},
            color=list(feature_importance.values()),
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit • Model: LightGBM • Accuracy: 79.5% Churn Detection</p>
    <p><small>For support contact: your-email@company.com</small></p>
</div>
""", unsafe_allow_html=True)

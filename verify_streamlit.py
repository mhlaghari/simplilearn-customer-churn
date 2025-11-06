import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def verify_streamlit_behavior():
    """Verify the exact behavior that should happen in Streamlit"""
    print("🔍 VERIFYING STREAMLIT BEHAVIOR")
    print("=" * 50)
    
    try:
        # Load model (same as Streamlit)
        model = joblib.load('best_churn_model_lightgbm.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Define exact feature order (MUST MATCH Streamlit app)
        FEATURE_ORDER = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender_encoded',
            'Geo_France', 'Geo_Germany', 'Geo_Spain'
        ]
        
        # Test the exact default values from Streamlit app
        default_customer = {
            'CreditScore': 650,
            'Age': 40,
            'Tenure': 5,
            'Balance': 50000.0,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 75000.0,
            'Gender_encoded': 1,
            'Geo_France': 1,
            'Geo_Germany': 0,
            'Geo_Spain': 0
        }
        
        print("Testing DEFAULT Streamlit values:")
        print(default_customer)
        
        # Create DataFrame with exact feature order
        df = pd.DataFrame([default_customer])[FEATURE_ORDER]
        
        # Scale numerical features
        numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        df_scaled = df.copy()
        df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
        
        print("\\n📊 Scaled features:")
        print(df_scaled.iloc[0][numerical_cols].to_dict())
        
        # Make prediction
        probabilities = model.predict_proba(df_scaled)[0]
        prediction = model.predict(df_scaled)[0]
        
        print(f"\\n🎯 PREDICTION RESULTS:")
        print(f"Probability of NO CHURN: {probabilities[0]:.3f}")
        print(f"Probability of CHURN: {probabilities[1]:.3f}")
        print(f"Prediction: {'CHURN' if prediction == 1 else 'NO CHURN'}")
        
        # Test edge cases
        print("\\n🧪 TESTING EDGE CASES:")
        
        test_cases = [
            {"name": "Very Low Risk", "changes": {"CreditScore": 800, "Age": 25, "NumOfProducts": 1}},
            {"name": "Very High Risk", "changes": {"CreditScore": 400, "Age": 60, "NumOfProducts": 4, "Geo_Germany": 1, "Geo_France": 0}}
        ]
        
        for test in test_cases:
            test_data = default_customer.copy()
            test_data.update(test["changes"])
            
            df_test = pd.DataFrame([test_data])[FEATURE_ORDER]
            df_test_scaled = df_test.copy()
            df_test_scaled[numerical_cols] = scaler.transform(df_test[numerical_cols])
            
            proba = model.predict_proba(df_test_scaled)[0][1]
            pred = model.predict(df_test_scaled)[0]
            
            print(f"{test['name']}: Churn Probability = {proba:.3f}, Prediction = {'CHURN' if pred == 1 else 'NO CHURN'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_streamlit_behavior()

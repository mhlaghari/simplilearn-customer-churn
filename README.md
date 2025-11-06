# 🏦 Bank Customer Churn Predictor

A professional Streamlit web application that predicts bank customer churn using Machine Learning with 79.5% accuracy.

## 🌐 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app/)

## 📊 Model Performance
- **Churn Detection Rate**: 79.5%
- **ROC-AUC Score**: 86.2%
- **Training Data**: 10,000 bank customers
- **Key Features**: Age, Balance, Credit Score, Products

## 🚀 Quick Deploy to Streamlit Cloud

### Method 1: One-Click Deploy
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=your-repo-url)

### Method 2: Manual Deploy
1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Click "New app"**
4. **Select your forked repository**
5. **Set Main file path to `app.py`**
6. **Click "Deploy"**

Your app will be live at: `https://yourusername-bank-churn-predictor-app-xxx.streamlit.app/`

## 📁 Files in this Repository
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `best_churn_model_lightgbm.pkl` - Trained machine learning model
- `scaler.pkl` - Feature scaling object
- `label_encoder.pkl` - Gender encoding object
- `README.md` - This file

## 💡 Features
- ✅ Single customer churn prediction
- ✅ Real-time risk assessment (High/Medium/Low)
- ✅ Personalized retention recommendations
- ✅ Feature importance visualization
- ✅ Mobile-responsive design
- ✅ Professional UI/UX

## 🎯 How to Use
1. Enter customer details in the form
2. Click "Predict Churn Probability"
3. View the churn risk assessment
4. Get specific retention recommendations
5. Understand which factors influenced the prediction

## 🔧 Technical Details
- **Framework**: Streamlit
- **ML Model**: LightGBM Classifier
- **Preprocessing**: StandardScaler, LabelEncoder
- **Visualization**: Matplotlib, Seaborn

## 📞 Support
For issues with deployment, please check:
1. All .pkl files are in the repository
2. requirements.txt has correct versions
3. Main file path is set to `app.py`

---

**Built with ❤️ for Bank Customer Retention**

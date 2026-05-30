
# 🔮 Customer Churn Prediction System

A production-ready machine learning pipeline for predicting customer churn using scikit-learn.

## 📊 Features
- **Advanced ML Pipeline**: Scikit-learn Pipeline API with preprocessing and RandomForest
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Web Interface**: Interactive Streamlit app
- **Batch Predictions**: CSV upload support
- **Model Performance**: ROC-AUC optimized

## 📈 Model Performance
- Accuracy: [ 0.7882]
- Precision: [0.6329]
- Recall: [0.4840]
- F1-Score: [ 0.5485]
- ROC-AUC:  0.8335]

## 🚀 Quick Start

### Local Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

### Online (Streamlit Cloud)
Visit:[ [Streamlit Cloud URL]](https://telco-churn-prediction-system.streamlit.app/)

## 📦 Requirements
- Python 3.8+
- scikit-learn 1.0+
- streamlit 1.0+
- pandas, numpy, joblib

## 🛠️ Technical Stack
- **Training**: Google Colab (GPU)
- **ML Framework**: scikit-learn
- **Web Framework**: Streamlit
- **Deployment**: Streamlit Cloud
- **Version Control**: GitHub

## 💡 Usage

### Single Prediction
Enter customer details and get instant churn probability.

### Batch Prediction
Upload CSV file with multiple customers for bulk predictions.

## 📝 Model Details
- **Algorithm**: Random Forest Classifier
- **Preprocessing**: StandardScaler + OneHotEncoder
- **Hyperparameter Tuning**: GridSearchCV with 5-fold CV
- **Data Split**: 80% train, 20% test

## 📚 Files
- `app.py`: Streamlit web application
- `churn_pipeline.pkl`: Trained ML pipeline (scikit-learn)
- `model_metadata.pkl`: Performance metrics and metadata
- `requirements.txt`: Python dependencies

## 🎓 Skills Demonstrated
- ✅ ML pipeline construction
- ✅ Hyperparameter tuning
- ✅ Model export and serialization
- ✅ Production-ready code
- ✅ Web application development
- ✅ CI/CD and deployment

## 📧 Contact
[Mahboob Ali] | [mahboobalilaghari1976@gmail.com]

---
Built for Advanced ML Task Set 2

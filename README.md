# ❤️ Heart Disease Prediction System

A machine learning-based web application that predicts the likelihood of heart disease using patient medical parameters.

The system uses **XGBoost** for prediction and **Streamlit** for the interactive web interface.

## 🚀 Features

- Predicts heart disease likelihood from patient information
- User-friendly Streamlit interface
- Handles missing values using preprocessing pipelines
- Numerical feature scaling
- Categorical feature encoding
- XGBoost classification model
- Displays prediction probability
- Shows model performance metrics

## 🧠 Machine Learning Workflow

The project follows this workflow:

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Missing Value Handling
4. Feature Preprocessing
5. Categorical Encoding
6. Feature Scaling
7. Model Training
8. Hyperparameter Tuning
9. Model Evaluation
10. Streamlit Deployment

## 🤖 Model

The final model used is **XGBoost Classifier**.

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 82.07% |
| Recall | 88.24% |
| F1 Score | 84.51% |
| ROC-AUC | 90.94% |

Recall was given particular importance because correctly identifying patients who may have heart disease is important for this educational prediction system.

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## 📂 Project Structure

```text
Heart_Disease_Project/
│
├── app.py
├── heart_preprocessor.pkl
├── heart_xgb_model.json
├── requirements.txt
└── README.md
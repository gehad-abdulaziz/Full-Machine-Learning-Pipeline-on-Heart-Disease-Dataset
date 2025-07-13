# Heart Disease Prediction Project

## Overview
This project aims to build a machine learning model to predict the risk of heart disease based on patient clinical data. It covers data preprocessing, feature selection, dimensionality reduction, model training, evaluation, and deployment of a Streamlit web app for user interaction.

## Dataset
- Dataset used: Cleveland Heart Disease Dataset (processed.cleveland.data)
- Number of features used: 13 clinical features plus target variable
- Source: UCI Machine Learning Repository

## How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
Run the Streamlit app


python -m streamlit run ui/app.py


Features Used
Age

Sex

Chest pain type (cp)

Resting blood pressure (trestbps)

Serum cholesterol (chol)

Fasting blood sugar (fbs)

Resting ECG results (restecg)

Maximum heart rate achieved (thalach)

Exercise induced angina (exang)

ST depression induced by exercise (oldpeak)

Slope of the peak exercise ST segment (slope)

Number of major vessels colored by fluoroscopy (ca)

Model
Pipeline includes data scaling, feature selection, PCA dimensionality reduction, and classification using Random Forest.

Achieved accuracy: ~62.4% (adjust based on your final results)

Precision, Recall, and F1 Score are detailed in results/evaluation_metrics.txt

Notes
The dataset is moderately imbalanced; SMOTE oversampling is applied during training.

Hyperparameter tuning is performed using GridSearchCV for best model parameters.
### The current model shows moderate accuracy but relatively low precision, recall, and F1 scores, especially for some classes. This indicates that the model may not be highly reliable for critical decision-making and should be improved further before deployment.


Author  :

Gehad Abdulaziz  

Email: gehadabdelaziz179@example.com  

GitHub: https://github.com/gehad-abdulaziz  
Linkedin: https://www.linkedin.com/in/gehad-abdulaziz-228973287/

### This project represents my very first experience in machine learning, and I’m excited to continue learning and improving.

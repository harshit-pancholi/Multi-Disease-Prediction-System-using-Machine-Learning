# Multi-Disease-Prediction-System-using-Machine-Learning
Multi-disease prediction system using Machine Learning that predicts heart disease, diabetes, and kidney disease. The project includes data preprocessing, model comparison, hyperparameter tuning, and a Streamlit web app for real-time predictions and visualization.

## Project Overview

  This project is a Machine Learning-based web application that predicts multiple diseases:
      * Heart Disease
      * Diabetes
      * Kidney Disease

The system uses trained ML models and provides real-time predictions through a Streamlit interface.

## Features

  * Multi-disease prediction
  * Hyperparameter tuning using GridSearchCV
  * Model comparison
  * Feature importance visualization
  * Prediction history tracking

## Tech Stack

  * Python
  * Scikit-learn
  * Pandas, NumPy
  * Streamlit
  * Matplotlib

## Project Structure

    datasets/       → Input datasets
    models/         → Trained ML models
    src/            → Training & prediction logic
    webapp/         → Streamlit application
    results/        → Output results


## How to Run

  ### 1. Install dependencies
    pip install -r requirements.txt
  
  ### 2. Run the app
    streamlit run webapp/app.py

## Models Used

  * Random Forest
  * Gradient Boosting
  * Support Vector Machine (SVM)
  * Logistic Regression


## Future Scope

  * Add more diseases
  * Improve model accuracy
  * Integrate with real-time healthcare systems

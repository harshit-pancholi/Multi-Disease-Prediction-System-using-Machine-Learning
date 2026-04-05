import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# CONFIGURATION
st.set_page_config(page_title="Medical AI Dashboard", layout="wide")

#LOAD MODELS
heart_model = pickle.load(open("models/heart_pipeline.pkl", "rb"))
diabetes_model = pickle.load(open("models/diabetes_pipeline.pkl", "rb"))
kidney_model = pickle.load(open("models/kidney_pipeline.pkl", "rb"))

#SAVE FUNCTION
def save_prediction(user, disease, features, inputs, prediction, probability):

    file_path = "results/prediction_history.csv"
    data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Name": user["name"],
        "Age": user["age"],
        "Gender": user["gender"],
        "Disease": disease,
        "Prediction": prediction,
        "Confidence": round(probability*100, 2) if probability else None
    }

    for f, val in zip(features, inputs):
        data[f] = val
    df = pd.DataFrame([data])

    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

#NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Disease Prediction", "Model Comparison"]
)

# HOME
if page == "Home":
    st.title("Multi Disease Prediction Dashboard")
    st.markdown("""
    AI-powered system for predicting:
    - ❤️ Heart Disease  
    - 🩸 Diabetes  
    - 🧪 Kidney Disease  
    """)

# PREDICTION
elif page == "Disease Prediction":
    st.title("Disease Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Name")
    with col2:
        age = st.slider("Age", 1, 100, 25)
    with col3:
        gender_ui = st.selectbox("Gender", ["Male", "Female"])
        gender = 1 if gender_ui == "Male" else 0

    user = {"name": name, "age": age, "gender": gender_ui}
    disease = st.selectbox(
        "Select Disease",
        ["Heart Disease", "Diabetes", "Kidney Disease"]
    )

    inputs = []
    feature_names = []

    # HEART DISEASE
    if disease == "Heart Disease":

        chest_pain = st.selectbox("Chest Pain Type", [0,1,2,3])
        bp = st.slider("Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 400, 200)
        fasting = 1 if st.selectbox("High Sugar?", ["No","Yes"])=="Yes" else 0
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        angina = 1 if st.selectbox("Exercise Pain?", ["No","Yes"])=="Yes" else 0
        oldpeak = st.slider("Heart Stress", 0.0, 6.0, 1.0)

        inputs = [age, gender, chest_pain, bp, chol,
                  fasting, max_hr, angina, oldpeak]
        feature_names = ["Age","Gender","Chest Pain","BP","Cholesterol",
                         "Sugar","Max HR","Angina","Stress"]

        model = heart_model

    # DIABETES 
    elif disease == "Diabetes":

        pregnancies = st.slider("Pregnancies", 0, 15, 1)
        glucose = st.slider("Glucose", 50, 200, 100)
        bp = st.slider("Blood Pressure", 60, 180, 80)
        skin = st.slider("Skin Thickness", 0, 100, 20)
        insulin = st.slider("Insulin", 0, 300, 80)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        dpf = st.slider("Family Risk", 0.0, 2.5, 0.5)

        inputs = [pregnancies, glucose, bp, skin,
                  insulin, bmi, dpf, age]
        feature_names = ["Pregnancies","Glucose","BP","Skin",
                         "Insulin","BMI","DPF","Age"]

        model = diabetes_model

    # KIDNEY DISEASE
    elif disease == "Kidney Disease":

        bp = st.slider("Blood Pressure", 80, 200, 120)
        sg = st.slider("Urine Density", 1.0, 1.03, 1.02)
        albumin = st.slider("Protein Level", 0, 5, 1)
        sugar = st.slider("Sugar Level", 0, 5, 0)
        glucose = st.slider("Blood Sugar", 50, 300, 120)
        urea = st.slider("Urea", 10, 200, 50)
        creatinine = st.slider("Creatinine", 0.5, 3.0, 1.2)
        hemo = st.slider("Hemoglobin", 5.0, 20.0, 12.0)

        inputs = [bp, sg, albumin, sugar,
                  glucose, urea, creatinine, hemo]
        feature_names = ["BP","SG","Albumin","Sugar",
                         "Glucose","Urea","Creatinine","Hemoglobin"]

        model = kidney_model

    # PREDICTION
    if st.button("🔍 Predict"):

        data = np.array(inputs).reshape(1, -1)

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.error("High Risk")
        else:
            st.success("Low Risk")

        if prob:
            st.progress(int(prob * 100))
            st.write(f"Confidence: {round(prob*100,2)}%")

        # SAVE (NO UI DISPLAY)
        save_prediction(user, disease, feature_names, inputs, prediction, prob)

        # EXPLAINABILITY 
        st.subheader("Key Factors")

        model_obj = model.named_steps["model"]

        if hasattr(model_obj, "feature_importances_"):
            importance = model_obj.feature_importances_
        elif hasattr(model_obj, "coef_"):
            importance = np.abs(model_obj.coef_[0])
        else:
            importance = np.ones(len(feature_names)) / len(feature_names)
            st.info("Feature importance not available for this model. Showing equal contribution.")
        importance = importance / np.sum(importance)

        fig, ax = plt.subplots()
        ax.barh(feature_names, importance)
        st.pyplot(fig)

        #SUGGESTIONS 
        st.subheader("💡 Suggestions")

        if prediction == 1:
            if disease == "Heart Disease":
                st.write("• Exercise regularly")
                st.write("• Reduce cholesterol intake")
            elif disease == "Diabetes":
                st.write("• Reduce sugar intake")
                st.write("• Monitor glucose regularly")
            else:
                st.write("• Stay hydrated")
                st.write("• Reduce salt intake")
        else:
            st.success("Maintain healthy lifestyle")

# MODEL COMPARISON
elif page == "Model Comparison":

    st.title("Model Comparison")
    df = pd.read_csv("results/model_comparison.csv")

    for disease in df["Disease"].unique():
        st.subheader(disease)

        subset = df[df["Disease"] == disease]
        st.dataframe(subset)

        st.bar_chart(subset.set_index("Model")["Accuracy"])
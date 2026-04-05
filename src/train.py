import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.utils import resample


# MODEL LIST
def get_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }


# TRAIN FUNCTION
def train_and_compare(X, y, disease_name, save_path):

    results = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model = None
    best_score = 0

    for name, model in get_models().items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # CV score
        cv_score = np.mean(cross_val_score(pipeline, X, y, cv=cv))

        print(f"\n{disease_name} - {name}")
        print("Accuracy:", acc)
        print("CV Score:", cv_score)

        results.append({
            "Disease": disease_name,
            "Model": name,
            "Accuracy": round(acc, 4),
            "CV Score": round(cv_score, 4)
        })
        # Best model selection
        if acc > best_score:
            best_score = acc
            best_model = pipeline
    # Save best model
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)
    return results


# HEART DISEASE
def train_heart():
    df = pd.read_csv("datasets/heart_disease_final.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    df = df.dropna(subset=['output'])
    df['output'] = df['output'].apply(lambda x: 1 if x == 1 else 0)
    features = ['Age','Gender','Chest pain type','Blood pressure(Normal)',
                'Cholesterol','Fasting blood pressure','Max Heartrate',
                'Exercise angina','ST depression']

    X = df[features]
    y = df['output']

    return train_and_compare(X, y, "Heart", "models/heart_pipeline.pkl")


# DIABETES 
def train_diabetes():
    df = pd.read_csv("datasets/diabetes_final.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    df = df.dropna(subset=['Outcome'])
    df['Outcome'] = df['Outcome'].apply(lambda x: 1 if x == 1 else 0)
    features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']

    X = df[features]
    y = df['Outcome']

    return train_and_compare(X, y, "Diabetes", "models/diabetes_pipeline.pkl")


# KIDNEY DISEASE
def train_kidney():
    df = pd.read_csv("datasets/kidney_final.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Handle labels
    if df['classification'].dtype == 'object':
        df['classification'] = df['classification'].astype(str).str.strip().str.lower()
        df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    else:
        df['classification'] = df['classification'].apply(lambda x: 1 if x == 1 else 0)

    df = df.dropna(subset=['classification'])

    # Handle imbalance
    df_majority = df[df['classification'] == 1]
    df_minority = df[df['classification'] == 0]

    if len(df_minority) > 0:
        df_minority = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        df = pd.concat([df_majority, df_minority])

    features = ['blood_pressure','Specific_Gravity','albumin','sugar',
                'blood_glucose_random','blood_urea','Serum_Creatinine','hemoglobin']

    X = df[features]
    y = df['classification']

    return train_and_compare(X, y, "Kidney", "models/kidney_pipeline.pkl")


# MAIN FUNCTION
if __name__ == "__main__":

    all_results = []

    all_results.extend(train_heart())
    all_results.extend(train_diabetes())
    all_results.extend(train_kidney())

    df = pd.DataFrame(all_results)

    # Save comparison
    df.to_csv("results/model_comparison.csv", index=False)
    print("\n✅ All models trained !!!")
    print(df)
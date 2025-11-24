import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os

df = pd.read_csv("enhanced_health_insurance_claims.csv")

np.random.seed(42)
df["fraud_label"] = np.random.choice([0, 1], size=len(df), p=[0.99, 0.01])

X = df.drop(columns=["fraud_label", "ClaimID", "PatientID", "ProviderID", "ClaimDate", "ProviderLocation", "DiagnosisCode", "ProcedureCode"], axis=1)
y = df["fraud_label"]


numeric_features = ["PatientAge", "ClaimAmount", "PatientIncome"]
categorical_features = [col for col in X.columns if X[col].dtype == 'object']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder='drop'
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "fraud_detection_model_enhanced.pkl")

print("Model saved as fraud_detection_model_enhanced.pkl")
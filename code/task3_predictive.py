# Task 3: Predictive Analytics - Random Forest Model
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists("../models"):
    os.makedirs("../models")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predict
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]  # probability for positive class

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Map probability to priority
def prob_to_priority(p):
    if p >= 0.85:
        return "low"
    elif p >= 0.60:
        return "medium"
    else:
        return "high"

priority = [prob_to_priority(p) for p in y_prob]
priority_counts = pd.Series(priority).value_counts()
print("\nPredicted Issue Priority Counts:\n", priority_counts)

# Save model and scaler
joblib.dump(rf, "../models/rf_model.joblib")
joblib.dump(scaler, "../models/scaler.joblib")
print("\nModel and scaler saved to /models directory.")

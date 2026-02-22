# ======================================
# Fake Currency Detection - Train Script
# ======================================

import pandas as pd
import numpy as np
import os
import joblib   # ✅ IMPORTANT (added)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --------------------------------------
# 1️⃣ Check Directory (for debugging)
# --------------------------------------

print("Current Directory:", os.getcwd())
print("Files here:", os.listdir())
print("Files in data folder:", os.listdir("data"))

# --------------------------------------
# 2️⃣ Load Dataset
# --------------------------------------

data = pd.read_csv("data/BankNote_Authentication.csv")

print("\nDataset Loaded Successfully!")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# --------------------------------------
# 3️⃣ Split Features & Target
# --------------------------------------

X = data.iloc[:, :-1]   # Features
y = data.iloc[:, -1]    # Target (class)

# --------------------------------------
# 4️⃣ Train-Test Split
# --------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------
# 5️⃣ Feature Scaling
# --------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------
# 6️⃣ Train Model
# --------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------
# 7️⃣ Prediction
# --------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------
# 8️⃣ Evaluation
# --------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", round(accuracy * 100, 2), "%")

# --------------------------------------
# 9️⃣ Save Model & Scaler
# --------------------------------------

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and Scaler saved successfully inside 'model' folder!")

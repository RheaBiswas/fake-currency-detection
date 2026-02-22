import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="centered"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #00c853;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# Load Dataset for Evaluation
# -----------------------------
data = pd.read_csv("data/BankNote_Authentication.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scale full dataset for proper accuracy
X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

# -----------------------------
# Title
# -----------------------------
st.title("💵 Fake Currency Detection System")
st.markdown("### Enter the banknote feature values below to check authenticity.")

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("📥 Input Features")

variance = st.number_input("Variance", format="%.5f")
skewness = st.number_input("Skewness", format="%.5f")
kurtosis = st.number_input("Kurtosis", format="%.5f")
entropy = st.number_input("Entropy", format="%.5f")

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    input_data = np.array([[variance, skewness, kurtosis, entropy]])

    # IMPORTANT: Scale input
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.markdown("---")

    if prediction == 0:
        st.success("✅ The Banknote is Authentic")
    else:
        st.error("❌ The Banknote is Fake")

# -----------------------------
# Model Performance Section
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Performance")

st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
st.subheader("📈 Confusion Matrix")

fig, ax = plt.subplots()

ax.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Authentic", "Fake"])
ax.set_yticklabels(["Authentic", "Fake"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.markdown("---")
st.caption("Built using Machine Learning & Streamlit 🚀")

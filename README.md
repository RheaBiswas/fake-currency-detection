# 💵 Fake Currency Detection System

A Machine Learning based web application that detects whether a banknote is **Authentic** or **Fake** using Logistic Regression.

Built using:
- Python
- Scikit-learn
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Joblib

---

## 📌 Project Overview

This project uses the **Banknote Authentication Dataset**.

The dataset contains the following input features:

- Variance of wavelet transformed image
- Skewness of wavelet transformed image
- Kurtosis of wavelet transformed image
- Entropy of image

Target Variable:
- `0` → Authentic Banknote
- `1` → Fake Banknote

The model is trained using **Logistic Regression** and achieves approximately **98% accuracy**.

---

## 📂 Project Structure

fake-currency-detection/
│
├── data/
│ └── BankNote_Authentication.csv
│
├── model/
│ ├── model.pkl
│ └── scaler.pkl
│
├── app.py
├── train.py
├── .gitignore
└── README.md


---

## ⚙️ How To Run This Project

### 1️⃣ Clone the Repository

git clone https://github.com/YOUR-USERNAME/fake-currency-detection.git

cd fake-currency-detection


---

### 2️⃣ Install Required Libraries

You can install manually:

Or create a virtual environment (recommended):

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


---

### 3️⃣ Train the Model

python train.py

This will:
- Load dataset
- Split data
- Scale features
- Train Logistic Regression model
- Save model & scaler inside `model/` folder

---

### 4️⃣ Run the Web Application

streamlit run app.py

The app will open in your browser.

---

## 📊 Model Performance

- Algorithm Used: Logistic Regression
- Accuracy: ~98%
- Confusion Matrix displayed in the app
- Feature Scaling using StandardScaler

---

## 🖥️ Application Features

✔ Enter banknote feature values  
✔ Real-time prediction  
✔ Model accuracy display  
✔ Confusion matrix visualization  
✔ Clean and modern UI  
✔ Scaler integrated for correct predictions  

---

## 🧠 Machine Learning Workflow

1. Data Loading  
2. Train-Test Split  
3. Feature Scaling  
4. Model Training  
5. Model Evaluation  
6. Model Saving  
7. Web Deployment using Streamlit  

---

## 🚀 Future Improvements

- Add Random Forest & SVM comparison
- Add model comparison dashboard
- Add CSV batch prediction upload
- Deploy on Streamlit Cloud
- Add accuracy graph & ROC curve

---

## 👩‍💻 Author

Rhea Biswas  
B.Tech AIML Student  

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub!



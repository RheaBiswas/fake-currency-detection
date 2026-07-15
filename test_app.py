import os
import joblib
import pandas as pd
import numpy as np

def test_model_pipeline():
    print("=== Testing ML Model Pipeline ===")
    
    # 1. Verify file paths exist
    model_path = "model/model.pkl"
    scaler_path = "model/scaler.pkl"
    data_path = "data/BankNote_Authentication.csv"
    
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert os.path.exists(scaler_path), f"Scaler not found at {scaler_path}"
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"
    print("[SUCCESS] All required files (model, scaler, data) exist.")
    
    # 2. Load assets
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    data = pd.read_csv(data_path)
    print("[SUCCESS] Loaded model, scaler, and dataset successfully.")
    
    # 3. Test dataset shape and columns
    expected_cols = ["variance", "skewness", "curtosis", "entropy", "class"]
    for col in expected_cols:
        assert col in data.columns, f"Expected column {col} missing from dataset"
    print(f"[SUCCESS] Dataset columns match expected features. Shape: {data.shape}")
    
    # 4. Test predictions on known samples
    # Sample authentic (class 0)
    auth_sample = data[data['class'] == 0].iloc[0]
    auth_features = pd.DataFrame([[
        auth_sample['variance'],
        auth_sample['skewness'],
        auth_sample['curtosis'],
        auth_sample['entropy']
    ]], columns=["variance", "skewness", "curtosis", "entropy"])
    
    # Scale input
    auth_scaled = scaler.transform(auth_features)
    auth_pred = model.predict(auth_scaled)[0]
    auth_probs = model.predict_proba(auth_scaled)[0]
    
    print(f"\nTesting Authentic Sample:")
    print(f"Features: {auth_features.values[0]}")
    print(f"Expected Class: 0 (Authentic)")
    print(f"Predicted Class: {auth_pred}")
    print(f"Probability [Authentic, Fake]: {auth_probs}")
    assert auth_pred == 0, "Failed prediction for authentic sample"
    print("[SUCCESS] Authentic sample tested successfully.")
    
    # Sample fake (class 1)
    fake_sample = data[data['class'] == 1].iloc[0]
    fake_features = pd.DataFrame([[
        fake_sample['variance'],
        fake_sample['skewness'],
        fake_sample['curtosis'],
        fake_sample['entropy']
    ]], columns=["variance", "skewness", "curtosis", "entropy"])
    
    # Scale input
    fake_scaled = scaler.transform(fake_features)
    fake_pred = model.predict(fake_scaled)[0]
    fake_probs = model.predict_proba(fake_scaled)[0]
    
    print(f"\nTesting Fake Sample:")
    print(f"Features: {fake_features.values[0]}")
    print(f"Expected Class: 1 (Fake)")
    print(f"Predicted Class: {fake_pred}")
    print(f"Probability [Authentic, Fake]: {fake_probs}")
    assert fake_pred == 1, "Failed prediction for fake sample"
    print("[SUCCESS] Fake sample tested successfully.")
    
    # 5. Full dataset evaluation accuracy check
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y, y_pred)
    print(f"\nFull dataset accuracy: {acc * 100:.2f}%")
    assert acc > 0.95, f"Model accuracy ({acc*100:.2f}%) is below 95% threshold"
    print("[SUCCESS] Model pipeline validation passed!")

if __name__ == "__main__":
    test_model_pipeline()

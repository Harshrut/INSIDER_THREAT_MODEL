import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shap

# === Load model, scaler, and encoder columns ===
model = joblib.load("app/model/isolation_forest_model.pkl")
scaler = joblib.load("app/model/scaler.pkl")
encoder_columns = joblib.load("app/model/encoder_columns.pkl")

# === SHAP explainer (initialize on small sample input) ===
# Dummy input for initializing explainer
dummy = np.zeros((100, len(encoder_columns)))
explainer = shap.Explainer(model, dummy)

# === FastAPI app ===
app = FastAPI()

# === Input schema ===
class InputData(BaseModel):
    hour: int
    day_of_week: str
    is_weekend: int
    is_off_hours: int
    tenure_months: int
    performance_rating: int
    salary_band: int
    session_duration: float
    failed_login_attempts: int
    files_accessed: int
    data_transfer_mb: float
    emails_sent: int
    usb_events: int
    print_jobs: int
    privilege_escalation_attempt: int
    vpn_connection: int
    unusual_app_usage: int
    policy_violation: int
    external_email_count: int
    security_alert_triggered: int
    source_ip_numeric: float
    department: str
    role: str
    access_level: str
    action: str
    device_type: str
    location: str

# === Risk score mapping ===
def compute_risk(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.full_like(scores, 5.0)
    return ((-scores - min_s) / (max_s - min_s) * 45 + 5).round(2)

# === Prediction endpoint ===
@app.post("/predict")
async def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])

        # One-hot encode categorical
        categorical_cols = [
            'day_of_week', 'department', 'role',
            'access_level', 'action', 'device_type', 'location'
        ]
        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        # Add missing columns
        for col in encoder_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[encoder_columns]

        # Scale
        df_scaled = scaler.transform(df_encoded)

        # Predict
        prediction = model.predict(df_scaled)[0]
        score = model.decision_function(df_scaled)[0]
        risk_score = compute_risk(np.array([score]))[0]

        # SHAP for recommendations
        shap_values = explainer(df_scaled)
        shap_vals = shap_values.values[0]
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_features = [(encoder_columns[i], float(shap_vals[i])) for i in top_idx]

        recommendations = [
            f"Investigate feature '{feat}' — SHAP impact: {val:.3f}"
            for feat, val in top_features
        ]

        return {
            "prediction": "Threat" if prediction == -1 else "Normal",
            "anomaly_score": float(score),
            "risk_score": float(risk_score),
            "top_features": top_features,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

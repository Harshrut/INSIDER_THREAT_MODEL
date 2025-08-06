from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load("app/model/isolation_forest_model.pkl")
scaler = joblib.load("app/model/scaler.pkl")

# Define request structure
class FeatureInput(BaseModel):
    features: list

@app.post("/predict")
def predict(input_data: FeatureInput):
    data = np.array(input_data.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return {"prediction": int(prediction[0])}

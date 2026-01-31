from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from feature_extraction import extract_features

app = FastAPI()

# Load once at startup
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

class SignalInput(BaseModel):
    voltage: list[float]
    current: list[float]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.post("/predict_swv", response_model=PredictionOutput)
def predict_swv(data: SignalInput):
    V = np.array(data.voltage, dtype=float)
    I = np.array(data.current, dtype=float)

    if len(V) < 10 or len(I) < 10:
        return {"prediction": -1, "probability": 0.0}

    # Extract features (dict)
    feats = extract_features(V, I)

    # Convert to ordered numeric vector
    X = np.array([feats[k] for k in feats], dtype=float).reshape(1, -1)

    # Scale
    Xs = scaler.transform(X)

    # Predict
    pred = int(model.predict(Xs)[0])
    prob = float(model.predict_proba(Xs)[0][1])

    return {
        "prediction": pred,
        "probability": prob
    }

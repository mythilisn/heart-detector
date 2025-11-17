# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

APP_PORT = 8000
MODEL_PATH = "models/xgb_heart_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train_heart.py first to produce models/xgb_heart_pipeline.joblib")

pipeline = joblib.load(MODEL_PATH)

# Try to get the feature names the pipeline expects
expected_cols = None
try:
    expected_cols = list(pipeline.named_steps['imputer'].feature_names_in_)
except Exception:
    # fallback: try scaler
    try:
        expected_cols = list(pipeline.named_steps['scaler'].feature_names_in_)
    except Exception:
        expected_cols = None

app = FastAPI(title="Heart Disease Risk Predictor")

class PatientRecord(BaseModel):
    data: dict

@app.post("/predict")
def predict(record: PatientRecord):
    # Create a DataFrame with expected columns in the right order
    input_dict = record.data

    if expected_cols is None:
        # If we couldn't retrieve expected columns, accept whatever user sent
        df = pd.DataFrame([input_dict])
    else:
        # Fill missing columns with NaN, extra columns are ignored
        rows = []
        row = []
        for c in expected_cols:
            row.append(input_dict.get(c, np.nan))
        df = pd.DataFrame([row], columns=expected_cols)

    try:
        proba = float(pipeline.predict_proba(df)[:, 1][0])
        pred = int(proba >= 0.5)
        return {"prediction": pred, "probability": proba, "features_used": list(df.columns)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Heart Disease Risk Predictor API. POST JSON to /predict with {'data': {...}}"}

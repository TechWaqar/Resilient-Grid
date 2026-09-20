from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import sys
import os


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# LOAD V7 MODEL PIPELINE
# ============================================================

from src.predict import (
    model,
    scaler,
    FEATURES,
    THRESHOLD
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Resilient-Grid API",
    description="AI-powered electrical anomaly detection API",
    version="1.0.0"
)


# ============================================================
# ELECTRICAL DATA SCHEMA
# ============================================================

class ElectricalData(BaseModel):

    # --------------------------------------------------------
    # Original sensor measurements
    # --------------------------------------------------------

    voltage: float
    current: float
    temperature: float
    power: float

    # --------------------------------------------------------
    # Change features
    # --------------------------------------------------------

    voltage_change: float
    current_change: float
    temperature_change: float
    power_change: float

    # --------------------------------------------------------
    # Causal rolling means
    # --------------------------------------------------------

    voltage_rolling_mean: float
    current_rolling_mean: float
    temperature_rolling_mean: float
    power_rolling_mean: float

    # --------------------------------------------------------
    # Causal rolling standard deviations
    # --------------------------------------------------------

    voltage_rolling_std: float
    current_rolling_std: float
    temperature_rolling_std: float
    power_rolling_std: float

    # --------------------------------------------------------
    # Deviation features
    # --------------------------------------------------------

    voltage_deviation: float
    current_deviation: float
    temperature_deviation: float
    power_deviation: float

    # --------------------------------------------------------
    # Percentage deviation features
    # --------------------------------------------------------

    voltage_deviation_pct: float
    current_deviation_pct: float
    temperature_deviation_pct: float
    power_deviation_pct: float


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "Resilient-Grid",
        "status": "online",
        "model": "Random Forest V7"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "Random Forest V7"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: ElectricalData):

    # Convert Pydantic model to dictionary
    values = data.model_dump()

    # Keep feature order exactly the same as training
    input_data = [
        values[feature]
        for feature in FEATURES
    ]

    # Create DataFrame
    X = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # Apply training scaler
    X_scaled = scaler.transform(X)

    # Get anomaly probability
    probability = float(
        model.predict_proba(
            X_scaled
        )[0][1]
    )

    # Apply decision threshold
    prediction = int(
        probability >= THRESHOLD
    )

    status = (
        "ANOMALY"
        if prediction == 1
        else "NORMAL"
    )

    # Return prediction
    return {

        "status": status,

        "prediction": prediction,

        "anomaly_probability": round(
            probability,
            3
        ),

        "threshold": THRESHOLD,

        "model": "Random Forest V7"
    }
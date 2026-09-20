import joblib
import pandas as pd


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "models/random_forest_v7.pkl"
SCALER_FILE = "models/scaler_v7.pkl"


# ============================================================
# V7 FEATURES
# ============================================================

FEATURES = [

    # Original sensor features
    "voltage",
    "current",
    "temperature",
    "power",

    # Change features
    "voltage_change",
    "current_change",
    "temperature_change",
    "power_change",

    # Causal rolling means
    "voltage_rolling_mean",
    "current_rolling_mean",
    "temperature_rolling_mean",
    "power_rolling_mean",

    # Causal rolling standard deviations
    "voltage_rolling_std",
    "current_rolling_std",
    "temperature_rolling_std",
    "power_rolling_std",

    # Deviation features
    "voltage_deviation",
    "current_deviation",
    "temperature_deviation",
    "power_deviation",

    # Percentage deviation features
    "voltage_deviation_pct",
    "current_deviation_pct",
    "temperature_deviation_pct",
    "power_deviation_pct",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Random Forest V7...")

model = joblib.load(MODEL_FILE)

print("Loading V7 scaler...")

scaler = joblib.load(SCALER_FILE)

print("V7 prediction pipeline ready.")


# ============================================================
# DECISION THRESHOLD
# ============================================================

THRESHOLD = 0.30


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_anomaly(data):

    input_data = pd.DataFrame(
        [[data[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    # Scale using the scaler fitted on training data
    input_scaled = scaler.transform(input_data)

    # Random Forest anomaly probability
    probability = float(
        model.predict_proba(input_scaled)[0][1]
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

    return {
        "status": status,
        "prediction": prediction,
        "anomaly_probability": round(
            probability,
            3
        )
    }
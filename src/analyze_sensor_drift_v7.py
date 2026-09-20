import joblib
import pandas as pd


# ============================================================
# FILES
# ============================================================

DATA_FILE = "data/electrical_features_v7.csv"
MODEL_FILE = "models/random_forest_v7.pkl"
SCALER_FILE = "models/scaler_v7.pkl"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "voltage",
    "current",
    "temperature",
    "power",

    "voltage_change",
    "current_change",
    "temperature_change",
    "power_change",

    "voltage_rolling_mean",
    "current_rolling_mean",
    "temperature_rolling_mean",
    "power_rolling_mean",

    "voltage_rolling_std",
    "current_rolling_std",
    "temperature_rolling_std",
    "power_rolling_std",

    "voltage_deviation",
    "current_deviation",
    "temperature_deviation",
    "power_deviation",

    "voltage_deviation_pct",
    "current_deviation_pct",
    "temperature_deviation_pct",
    "power_deviation_pct"
]


THRESHOLD = 0.30


# ============================================================
# LOAD DATA
# ============================================================

print("Loading V7 feature dataset...")

df = pd.read_csv(DATA_FILE)

split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

print(f"Test rows: {len(test_df)}")


# ============================================================
# LOAD MODEL + SCALER
# ============================================================

print("\nLoading V7 model...")

model = joblib.load(MODEL_FILE)

print("Loading V7 scaler...")

scaler = joblib.load(SCALER_FILE)


# ============================================================
# PREDICT
# ============================================================

X_test = test_df[FEATURES]

X_scaled = scaler.transform(X_test)

probabilities = model.predict_proba(
    X_scaled
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


test_df["anomaly_probability"] = probabilities
test_df["prediction"] = predictions


# ============================================================
# SENSOR DRIFT
# ============================================================

sensor_drift = test_df[
    test_df["fault_type"] == "sensor_drift"
].copy()


print("\n" + "=" * 80)
print("SENSOR DRIFT ANALYSIS")
print("=" * 80)

print(
    f"Total sensor-drift samples: "
    f"{len(sensor_drift)}"
)

print(
    f"Detected: "
    f"{(sensor_drift['prediction'] == 1).sum()}"
)

print(
    f"Missed: "
    f"{(sensor_drift['prediction'] == 0).sum()}"
)


# ============================================================
# PROBABILITY STATISTICS
# ============================================================

print("\nAnomaly probability statistics:")

print(
    sensor_drift["anomaly_probability"]
    .describe()
)


# ============================================================
# DETECTED VS MISSED
# ============================================================

detected = sensor_drift[
    sensor_drift["prediction"] == 1
]

missed = sensor_drift[
    sensor_drift["prediction"] == 0
]


# ============================================================
# FEATURE COMPARISON
# ============================================================

comparison_features = [
    "voltage",
    "current",
    "temperature",
    "power",
    "voltage_change",
    "current_change",
    "temperature_change",
    "power_change",
    "voltage_deviation",
    "current_deviation",
    "temperature_deviation",
    "power_deviation",
    "voltage_deviation_pct",
    "current_deviation_pct",
    "temperature_deviation_pct",
    "power_deviation_pct"
]


print("\n" + "=" * 80)
print("DETECTED VS MISSED SENSOR DRIFT")
print("=" * 80)

comparison = pd.DataFrame({
    "detected_mean": detected[
        comparison_features
    ].mean(),

    "missed_mean": missed[
        comparison_features
    ].mean()
})


print(comparison.to_string())


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = "data/sensor_drift_analysis_v7.csv"

comparison.to_csv(
    OUTPUT_FILE
)

print("\nAnalysis saved to:")
print(OUTPUT_FILE)
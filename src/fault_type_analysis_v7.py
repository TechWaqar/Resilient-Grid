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


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.30


# ============================================================
# LOAD DATA
# ============================================================

print("Loading V7 feature dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Total rows: {len(df)}")


# ============================================================
# CHRONOLOGICAL 80/20 SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(f"Train rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")


# ============================================================
# TEST DATA
# ============================================================

X_test = test_df[FEATURES]

y_test = test_df["anomaly"]


print(f"Test normal: {(y_test == 0).sum()}")
print(f"Test anomaly: {(y_test == 1).sum()}")


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

print("\nLoading Random Forest V7...")

model = joblib.load(MODEL_FILE)

print("V7 model loaded.")

print("Loading V7 scaler...")

scaler = joblib.load(SCALER_FILE)

print("V7 scaler loaded.")


# ============================================================
# SCALE TEST DATA
# ============================================================

print("\nScaling test data...")

X_test_scaled = scaler.transform(X_test)

print("Scaling complete.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating V7 predictions...")

probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# ADD PREDICTIONS TO TEST DATA
# ============================================================

test_results = test_df.copy()

test_results["anomaly_probability"] = probabilities

test_results["prediction"] = predictions


# ============================================================
# ONLY REAL ANOMALIES
# ============================================================

anomalies = test_results[
    test_results["anomaly"] == 1
].copy()


print(
    f"\nAnalyzing {len(anomalies)} "
    "known anomalous test samples..."
)


# ============================================================
# FAULT TYPE ANALYSIS
# ============================================================

results = []


for fault_type, group in anomalies.groupby(
    "fault_type"
):

    total = len(group)

    detected = (
        group["prediction"] == 1
    ).sum()

    missed = (
        group["prediction"] == 0
    ).sum()

    detection_rate = (
        detected / total
        if total > 0
        else 0
    )

    results.append({
        "fault_type": fault_type,
        "total_samples": total,
        "detected": detected,
        "missed": missed,
        "detection_rate": detection_rate
    })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# Sort by detection rate
results_df = results_df.sort_values(
    "detection_rate",
    ascending=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("V7 FAULT-TYPE DETECTION ANALYSIS")
print("=" * 80)

print(
    f"{'Fault Type':<25}"
    f"{'Total':<10}"
    f"{'Detected':<12}"
    f"{'Missed':<10}"
    f"{'Detection Rate':<15}"
)

print("-" * 80)


for _, row in results_df.iterrows():

    print(
        f"{str(row['fault_type']):<25}"
        f"{int(row['total_samples']):<10}"
        f"{int(row['detected']):<12}"
        f"{int(row['missed']):<10}"
        f"{row['detection_rate']:.3f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = "data/fault_type_analysis_v7.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# OVERALL RESULT
# ============================================================

total_anomalies = len(anomalies)

total_detected = (
    anomalies["prediction"] == 1
).sum()

overall_detection_rate = (
    total_detected / total_anomalies
    if total_anomalies > 0
    else 0
)


print("\n" + "=" * 80)
print("OVERALL V7 ANOMALY DETECTION")
print("=" * 80)

print(f"Total anomalous samples: {total_anomalies}")
print(f"Detected anomalous samples: {total_detected}")
print(
    f"Overall detection rate: "
    f"{overall_detection_rate:.3f}"
)

print("\nResults saved to:")
print(OUTPUT_FILE)
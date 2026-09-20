import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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
# PREPARE TEST DATA
# ============================================================

X_test = test_df[FEATURES]
y_test = test_df["anomaly"]

print(f"Test normal: {(y_test == 0).sum()}")
print(f"Test anomaly: {(y_test == 1).sum()}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Random Forest V7...")

model = joblib.load(MODEL_FILE)

print("V7 model loaded.")


# ============================================================
# LOAD SCALER
# ============================================================

print("Loading V7 scaler...")

scaler = joblib.load(SCALER_FILE)

print("V7 scaler loaded.")


# ============================================================
# SCALE TEST DATA
# ============================================================

print("\nApplying V7 scaler to test data...")

X_test_scaled = scaler.transform(X_test)

print("Scaling complete.")


# ============================================================
# GET ANOMALY PROBABILITIES
# ============================================================

print("\nGenerating anomaly probabilities...")

probabilities = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


results = []


print("\n" + "=" * 85)
print("V7 THRESHOLD ANALYSIS")
print("=" * 85)

print(
    f"{'Threshold':<12}"
    f"{'TN':<8}"
    f"{'FP':<8}"
    f"{'FN':<8}"
    f"{'TP':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
    f"{'F1':<10}"
)

print("-" * 85)


for threshold in thresholds:

    # Convert probabilities into predictions
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Precision
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    # Recall
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    # F1
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # Store results
    results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    # Print results
    print(
        f"{threshold:<12.2f}"
        f"{tn:<8}"
        f"{fp:<8}"
        f"{fn:<8}"
        f"{tp:<8}"
        f"{precision:<12.3f}"
        f"{recall:<10.3f}"
        f"{f1:<10.3f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_FILE = "data/threshold_analysis_v7.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 85)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 85)

print(f"Saved to: {OUTPUT_FILE}")

print("\nV7 threshold analysis saved successfully.")
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# FILES
# ============================================================

INPUT_FILE = "data/electrical_features_v7.csv"

MODEL_FILE = "models/random_forest_v7.pkl"
SCALER_FILE = "models/scaler_v7.pkl"

COMPARISON_FILE = "data/model_comparison_v7.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading V7 feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df)}")


# ============================================================
# FEATURES
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

    # Deviations
    "voltage_deviation",
    "current_deviation",
    "temperature_deviation",
    "power_deviation",

    # Percentage deviations
    "voltage_deviation_pct",
    "current_deviation_pct",
    "temperature_deviation_pct",
    "power_deviation_pct",
]


TARGET = "anomaly"


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# CHRONOLOGICAL 80/20 SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nChronological split:")
print(f"Training rows: {len(train_df)}")
print(f"Testing rows: {len(test_df)}")


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nTraining anomaly distribution:")
print(train_df[TARGET].value_counts())

print("\nTesting anomaly distribution:")
print(test_df[TARGET].value_counts())


# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# SCALE FEATURES
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\nFitting StandardScaler on training data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest V7...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

print("\nEvaluating V7 model...")

y_pred = model.predict(X_test_scaled)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST V7 RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("\nModel saved:")
print(MODEL_FILE)

print("\nScaler saved:")
print(SCALER_FILE)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

comparison = pd.DataFrame([
    {
        "model": "Random Forest V7",
        "features": len(FEATURES),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
])

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)

print("\nComparison file saved:")
print(COMPARISON_FILE)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nTop 15 Feature Importances:")
print(
    importance_df
    .head(15)
    .to_string(index=False)
)

print("\nV7 training complete!")
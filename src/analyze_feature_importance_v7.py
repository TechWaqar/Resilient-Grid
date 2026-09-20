import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/random_forest_v7.pkl"
OUTPUT_FILE = "data/feature_importance_v7.csv"


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
    "power_deviation_pct",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Random Forest V7...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_


if len(importance) != len(FEATURES):
    raise ValueError(
        f"Feature count mismatch: "
        f"model has {len(importance)} importances "
        f"but FEATURES contains {len(FEATURES)} features."
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": importance
    }
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


df["rank"] = df.index + 1


# ============================================================
# REORDER COLUMNS
# ============================================================

df = df[
    [
        "rank",
        "feature",
        "importance"
    ]
]


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("RANDOM FOREST V7 FEATURE IMPORTANCE")
print("=" * 75)

for _, row in df.iterrows():

    print(
        f"{int(row['rank']):2d}. "
        f"{row['feature']:<30} "
        f"{row['importance']:.6f}"
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 75)
print("Analysis saved to:")
print(OUTPUT_FILE)
print("=" * 75)
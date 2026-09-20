import pandas as pd
import numpy as np


INPUT_FILE = "data/electrical_data_v3.csv"
OUTPUT_FILE = "data/electrical_features_v7.csv"


print("Loading V3 dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df)}")


# ============================================================
# TIME FEATURE
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour


# ============================================================
# CHANGE FEATURES
# ============================================================

df["voltage_change"] = df["voltage"].diff()

df["current_change"] = df["current"].diff()

df["temperature_change"] = df["temperature"].diff()

df["power_change"] = df["power"].diff()


# ============================================================
# CAUSAL ROLLING BASELINE
# IMPORTANT:
# Shift by 1 so the current reading is NOT included
# in its own rolling baseline.
# ============================================================

window = 10


df["voltage_rolling_mean"] = (
    df["voltage"]
    .shift(1)
    .rolling(window)
    .mean()
)

df["current_rolling_mean"] = (
    df["current"]
    .shift(1)
    .rolling(window)
    .mean()
)

df["temperature_rolling_mean"] = (
    df["temperature"]
    .shift(1)
    .rolling(window)
    .mean()
)

df["power_rolling_mean"] = (
    df["power"]
    .shift(1)
    .rolling(window)
    .mean()
)


# ============================================================
# CAUSAL ROLLING STANDARD DEVIATION
# ============================================================

df["voltage_rolling_std"] = (
    df["voltage"]
    .shift(1)
    .rolling(window)
    .std()
)

df["current_rolling_std"] = (
    df["current"]
    .shift(1)
    .rolling(window)
    .std()
)

df["temperature_rolling_std"] = (
    df["temperature"]
    .shift(1)
    .rolling(window)
    .std()
)

df["power_rolling_std"] = (
    df["power"]
    .shift(1)
    .rolling(window)
    .std()
)


# ============================================================
# DEVIATION FROM PREVIOUS BASELINE
# ============================================================

df["voltage_deviation"] = (
    df["voltage"]
    - df["voltage_rolling_mean"]
)

df["current_deviation"] = (
    df["current"]
    - df["current_rolling_mean"]
)

df["temperature_deviation"] = (
    df["temperature"]
    - df["temperature_rolling_mean"]
)

df["power_deviation"] = (
    df["power"]
    - df["power_rolling_mean"]
)


# ============================================================
# PERCENTAGE DEVIATION
# ============================================================

df["voltage_deviation_pct"] = (
    df["voltage_deviation"]
    / df["voltage_rolling_mean"]
)

df["current_deviation_pct"] = (
    df["current_deviation"]
    / df["current_rolling_mean"]
)

df["temperature_deviation_pct"] = (
    df["temperature_deviation"]
    / df["temperature_rolling_mean"]
)

df["power_deviation_pct"] = (
    df["power_deviation"]
    / df["power_rolling_mean"]
)


# ============================================================
# REMOVE INVALID INITIAL ROWS
# ============================================================

df = df.dropna().reset_index(drop=True)


# ============================================================
# SAVE V7 FEATURES
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# INFORMATION
# ============================================================

print(f"Final rows: {len(df)}")

print("\nV7 features created:")

print("Original features:")
print("voltage")
print("current")
print("temperature")
print("power")

print("\nChange features:")
print("voltage_change")
print("current_change")
print("temperature_change")
print("power_change")

print("\nCausal rolling features:")
print("voltage_rolling_mean")
print("current_rolling_mean")
print("temperature_rolling_mean")
print("power_rolling_mean")

print("voltage_rolling_std")
print("current_rolling_std")
print("temperature_rolling_std")
print("power_rolling_std")

print("\nDeviation features:")
print("voltage_deviation")
print("current_deviation")
print("temperature_deviation")
print("power_deviation")

print("\nPercentage deviation features:")
print("voltage_deviation_pct")
print("current_deviation_pct")
print("temperature_deviation_pct")
print("power_deviation_pct")

print(
    f"\nSaved to: {OUTPUT_FILE}"
)
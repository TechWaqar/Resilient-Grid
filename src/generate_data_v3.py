import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 12000

timestamps = pd.date_range(
    start="2026-01-01",
    periods=N,
    freq="min"
)

# ============================================================
# NORMAL ELECTRICAL BEHAVIOR
# ============================================================

voltage = np.random.normal(230, 4, N)
current = np.random.normal(10, 1.5, N)
temperature = np.random.normal(35, 3, N)

# Natural operating variation
time = np.arange(N)

voltage += 2 * np.sin(time / 300)
current += 0.8 * np.sin(time / 200)
temperature += 1.5 * np.sin(time / 500)

# ============================================================
# INITIALIZE LABELS
# ============================================================

anomaly = np.zeros(N, dtype=int)
fault_type = np.array(["normal"] * N, dtype=object)


# ============================================================
# FAULT 1: VOLTAGE DEGRADATION
# Two occurrences: training + future test
# Gradual mild -> severe degradation
# ============================================================

# First occurrence
start = 1200
end = 1260

voltage[start:end] -= np.linspace(5, 35, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "voltage_degradation"


# Second occurrence
start = 7200
end = 7260

voltage[start:end] -= np.linspace(5, 35, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "voltage_degradation"


# ============================================================
# FAULT 2: CURRENT SPIKE
# Two occurrences
# ============================================================

# First occurrence
start = 2500
end = 2560

current[start:end] += np.linspace(3, 8, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "current_spike"


# Second occurrence
start = 9000
end = 9060

current[start:end] += np.linspace(3, 8, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "current_spike"


# ============================================================
# FAULT 3: OVERHEATING
# Two occurrences
# ============================================================

# First occurrence
start = 3800
end = 3880

temperature[start:end] += np.linspace(5, 18, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "overheating"


# Second occurrence
start = 10000
end = 10080

temperature[start:end] += np.linspace(5, 18, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "overheating"


# ============================================================
# FAULT 4: COMBINED FAULT
# Two occurrences
# ============================================================

# First occurrence
start = 5200
end = 5270

voltage[start:end] -= 8
current[start:end] += 4
temperature[start:end] += 10

anomaly[start:end] = 1
fault_type[start:end] = "combined_fault"


# Second occurrence
start = 10800
end = 10870

voltage[start:end] -= 8
current[start:end] += 4
temperature[start:end] += 10

anomaly[start:end] = 1
fault_type[start:end] = "combined_fault"


# ============================================================
# FAULT 5: ELECTRICAL DISTURBANCE
# Two occurrences
# ============================================================

# First occurrence
start = 6800
end = 6820

voltage[start:end] -= 15
current[start:end] += 5

anomaly[start:end] = 1
fault_type[start:end] = "electrical_disturbance"


# Second occurrence
start = 11500
end = 11520

voltage[start:end] -= 15
current[start:end] += 5

anomaly[start:end] = 1
fault_type[start:end] = "electrical_disturbance"


# ============================================================
# FAULT 6: SENSOR DRIFT
# Two occurrences
# ============================================================

# First occurrence
start = 8200
end = 8300

temperature[start:end] += np.linspace(0, 12, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "sensor_drift"


# Second occurrence
start = 11000
end = 11100

temperature[start:end] += np.linspace(0, 12, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "sensor_drift"


# ============================================================
# FAULT 7: POWER ABNORMALITY
# Two occurrences
# ============================================================

# First occurrence
start = 9500
end = 9580

current[start:end] += np.linspace(2, 5, end - start)
voltage[start:end] -= np.linspace(2, 6, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "power_abnormality"


# Second occurrence
start = 11800
end = 11880

current[start:end] += np.linspace(2, 5, end - start)
voltage[start:end] -= np.linspace(2, 6, end - start)

anomaly[start:end] = 1
fault_type[start:end] = "power_abnormality"


# ============================================================
# CALCULATE POWER
# ============================================================

power = voltage * current


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "timestamp": timestamps,
    "voltage": voltage,
    "current": current,
    "temperature": temperature,
    "power": power,
    "anomaly": anomaly,
    "fault_type": fault_type
})


# ============================================================
# SAVE DATASET
# ============================================================

output_path = Path("data/electrical_data_v3.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("Realistic dataset V3 created!")
print(f"Rows: {len(df)}")
print(f"Normal samples: {(anomaly == 0).sum()}")
print(f"Anomaly samples: {(anomaly == 1).sum()}")

print("\nAnomaly percentage:")
print(f"{anomaly.mean() * 100:.2f}%")

print("\nFault distribution:")
print(df["fault_type"].value_counts())

print("\nVoltage degradation examples:")
print(
    df[df["fault_type"] == "voltage_degradation"]
    [["timestamp", "voltage", "current", "temperature", "power"]]
    .head(10)
)

print("\nVoltage degradation statistics:")
print(
    df[df["fault_type"] == "voltage_degradation"]
    [["voltage", "current", "temperature", "power"]]
    .describe()
)

print(f"\nSaved to: {output_path}")
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Load V3 dataset
input_path = Path("data/electrical_data_v3.csv")
df = pd.read_csv(input_path)

print("V3 dataset loaded!")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

features = [
    "voltage",
    "current",
    "temperature",
    "power"
]

X = df[features]
y = df["anomaly"]

# 80/20 time-based split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nTime-based split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print(f"\nTraining anomalies: {y_train.sum()}")
print(f"Testing anomalies: {y_test.sum()}")

# Fit scaler ONLY on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=features
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=features
)

# Add labels
X_train_scaled["anomaly"] = y_train.values
X_test_scaled["anomaly"] = y_test.values

# Keep fault type for later error analysis
X_train_scaled["fault_type"] = df["fault_type"].iloc[:split_index].values
X_test_scaled["fault_type"] = df["fault_type"].iloc[split_index:].values

# Save prepared datasets
train_path = Path("data/train_v3.csv")
test_path = Path("data/test_v3.csv")

X_train_scaled.to_csv(train_path, index=False)
X_test_scaled.to_csv(test_path, index=False)

# Save scaler
joblib.dump(scaler, "models/scaler_v3.pkl")

print("\nPreparation completed!")
print(f"Training data saved to: {train_path}")
print(f"Testing data saved to: {test_path}")
print("Scaler saved to: models/scaler_v3.pkl")

print("\nTest fault distribution:")
print(X_test_scaled["fault_type"].value_counts())
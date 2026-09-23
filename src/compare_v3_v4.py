import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv("data/test_v3.csv")

features = [
    "voltage",
    "current",
    "temperature",
    "power"
]

X_test = test_df[features]
y_test = test_df["anomaly"]

# ============================================================
# LOAD MODELS
# ============================================================

v3_model = joblib.load(
    "models/isolation_forest_v3.pkl"
)

v4_model = joblib.load(
    "models/isolation_forest_v4.pkl"
)

rf_model = joblib.load(
    "models/random_forest_v3.pkl"
)

# ============================================================
# PREDICTIONS
# ============================================================

v3_prediction = (
    v3_model.predict(X_test) == -1
).astype(int)

v4_prediction = (
    v4_model.predict(X_test) == -1
).astype(int)

rf_prediction = rf_model.predict(X_test)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, predictions):

    return {
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, predictions), 3
        ),
        "Precision": round(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            3
        ),
        "Recall": round(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            3
        ),
        "F1 Score": round(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
            3
        )
    }


# ============================================================
# CREATE COMPARISON
# ============================================================

results = [

    evaluate_model(
        "Isolation Forest V3 - Mixed Training",
        v3_prediction
    ),

    evaluate_model(
        "Isolation Forest V4 - Normal Only",
        v4_prediction
    ),

    evaluate_model(
        "Random Forest V3",
        rf_prediction
    )
]

results_df = pd.DataFrame(results)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 90)
print("RESILIENT-GRID MODEL EXPERIMENT COMPARISON")
print("=" * 90)

print("\n")
print(results_df.to_string(index=False))

# ============================================================
# SAVE RESULTS
# ============================================================

output_path = "data/model_experiment_comparison.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\nComparison saved to:")
print(output_path)

print("\n" + "=" * 90)
print("EXPERIMENT COMPARISON COMPLETED!")
print("=" * 90)


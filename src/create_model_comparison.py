import pandas as pd


# ============================================================
# MODEL COMPARISON
# ============================================================

data = [
    {
        "model": "V3 Random Forest",
        "features": 4,
        "accuracy": 0.926,
        "precision": 0.828,
        "recall": 0.620,
        "f1": 0.709,
        "notes": "Original sensor features"
    },
    {
        "model": "V4 Isolation Forest",
        "features": 4,
        "accuracy": 0.884,
        "precision": 0.631,
        "recall": 0.494,
        "f1": 0.554,
        "notes": "Normal-only training"
    },
    {
        "model": "V5 Random Forest",
        "features": 16,
        "accuracy": 0.960,
        "precision": 0.911,
        "recall": 0.789,
        "f1": 0.845,
        "notes": "Feature engineering"
    },
    {
        "model": "V6 Random Forest",
        "features": 24,
        "accuracy": 0.965,
        "precision": 0.962,
        "recall": 0.791,
        "f1": 0.868,
        "notes": "Extended deviation features"
    },
    {
        "model": "V7 Random Forest",
        "features": 24,
        "accuracy": 0.967,
        "precision": 0.979,
        "recall": 0.794,
        "f1": 0.877,
        "notes": "Causal rolling features"
    }
]


df = pd.DataFrame(data)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 95)
print("RESILIENT-GRID MODEL DEVELOPMENT COMPARISON")
print("=" * 95)

print(df.to_string(index=False))


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = "data/model_development_comparison.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)
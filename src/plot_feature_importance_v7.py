import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

INPUT_FILE = "data/feature_importance_v7.csv"
OUTPUT_FILE = "data/feature_importance_v7.png"

df = pd.read_csv(INPUT_FILE)


# ============================================================
# SELECT TOP FEATURES
# ============================================================

top_features = df.head(15).copy()

top_features = top_features.sort_values(
    "importance",
    ascending=True
)


# ============================================================
# CREATE CHART
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title(
    "Random Forest V7 — Top 15 Feature Importances"
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nChart saved to:")
print(OUTPUT_FILE)

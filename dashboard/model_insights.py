import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resilient-Grid Model Insights",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Resilient Grid AI Model Insights")

st.write(
    "Technical overview of the Random Forest V7 anomaly "
    "detection model."
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    "96.7%"
)

col2.metric(
    "Precision",
    "97.9%"
)

col3.metric(
    "Recall",
    "79.4%"
)

col4.metric(
    "F1 Score",
    "87.7%"
)


st.caption(
    "Evaluation performed on the chronological V7 test set."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("Model Configuration")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("**Algorithm**")
    st.write("Random Forest Classifier")

    st.write("**Number of trees**")
    st.write("300")

    st.write("**Feature count**")
    st.write("24")

    st.write("**Class weighting**")
    st.write("Balanced")


with info_col2:

    st.write("**Train | test strategy**")
    st.write("Chronological 80/20 split")

    st.write("**Feature scaling**")
    st.write("StandardScaler fitted on training data")

    st.write("**Decision threshold**")
    st.write("0.30")

    st.write("**Rolling features**")
    st.write("Previous readings only")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("Top V7 Feature Importances")

try:

    importance_df = pd.read_csv(
        "data/feature_importance_v7.csv"
    )

    top_features = importance_df.head(15).copy()

    chart_df = top_features.set_index(
        "feature"
    )["importance"]

    st.bar_chart(
        chart_df,
        horizontal=True
    )

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

except FileNotFoundError:

    st.error(
        "Feature importance file was not found."
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.header("What the Model Is Learning")

st.write(
    """
The V7 model combines raw electrical measurements with
temporal features describing the recent operating baseline.

The strongest features include rolling temperature, power,
voltage, and current measurements. This indicates that the
model is using both the current electrical state and its
relationship to recent operating conditions.
"""
)


# ============================================================
# CAUSAL FEATURE ENGINEERING
# ============================================================

st.header("Why V7 Uses Causal Rolling Features")

st.write(
    """
Rolling statistics in V7 are calculated from previous
readings rather than including the current observation.

This better represents a real-time monitoring scenario:

Previous sensor readings
        ↓
Recent operating baseline
        ↓
Current sensor reading
        ↓
Deviation from baseline
        ↓
Random Forest V7
        ↓
Normal / Anomaly
"""
)


# ============================================================
# MODEL LIMITATION
# ============================================================

st.header("Current Model Limitation")

st.warning(
    """
Subtle sensor drift remains challenging. In the V7 test set,
51 of 100 sensor-drift samples were detected at the 0.30
decision threshold.

The model detects stronger electrical abnormalities more
reliably than subtle drift patterns.
"""
)


# ============================================================
# FAULT DETECTION SUMMARY
# ============================================================

st.header("Fault-Type Detection")

fault_data = pd.DataFrame(
    {
        "Fault Type": [
            "Combined fault",
            "Electrical disturbance",
            "Overheating",
            "Power abnormality",
            "Sensor drift"
        ],
        "Detection Rate": [
            1.000,
            1.000,
            0.950,
            0.887,
            0.510
        ]
    }
)

fault_data["Detection Rate"] = (
    fault_data["Detection Rate"] * 100
).round(1)

st.dataframe(
    fault_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Resilient-Grid | Random Forest V7 | "
    "Edge-oriented electrical anomaly detection research prototype"
)
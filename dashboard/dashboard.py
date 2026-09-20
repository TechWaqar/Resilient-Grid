import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resilient-Grid AI Monitor",
    page_icon="⚡",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚡ Resilient-Grid AI Monitor")

st.write(
    "AI-powered electrical anomaly detection using "
    "Random Forest V7."
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# SENSOR INPUTS
# ============================================================

st.header("Electrical Sensor Readings")

voltage = st.number_input(
    "Voltage (V)",
    min_value=0.0,
    max_value=500.0,
    value=230.0
)

current = st.number_input(
    "Current (A)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=150.0,
    value=35.0
)

power = st.number_input(
    "Power (W)",
    min_value=0.0,
    max_value=50000.0,
    value=2300.0
)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

st.header("Temporal Features")

voltage_change = st.number_input(
    "Voltage Change",
    value=0.0
)

current_change = st.number_input(
    "Current Change",
    value=0.0
)

temperature_change = st.number_input(
    "Temperature Change",
    value=0.0
)

power_change = st.number_input(
    "Power Change",
    value=0.0
)


# ============================================================
# ROLLING FEATURES
# ============================================================

st.header("Rolling Statistics")

voltage_rolling_mean = st.number_input(
    "Voltage Rolling Mean",
    value=230.0
)

current_rolling_mean = st.number_input(
    "Current Rolling Mean",
    value=10.0
)

temperature_rolling_mean = st.number_input(
    "Temperature Rolling Mean",
    value=35.0
)

power_rolling_mean = st.number_input(
    "Power Rolling Mean",
    value=2300.0
)

voltage_rolling_std = st.number_input(
    "Voltage Rolling Std",
    value=2.0
)

current_rolling_std = st.number_input(
    "Current Rolling Std",
    value=1.0
)

temperature_rolling_std = st.number_input(
    "Temperature Rolling Std",
    value=2.0
)

power_rolling_std = st.number_input(
    "Power Rolling Std",
    value=100.0
)


# ============================================================
# DEVIATION FEATURES
# ============================================================

st.header("Deviation Features")

voltage_deviation = st.number_input(
    "Voltage Deviation",
    value=0.0
)

current_deviation = st.number_input(
    "Current Deviation",
    value=0.0
)

temperature_deviation = st.number_input(
    "Temperature Deviation",
    value=0.0
)

power_deviation = st.number_input(
    "Power Deviation",
    value=0.0
)

voltage_deviation_pct = st.number_input(
    "Voltage Deviation %",
    value=0.0
)

current_deviation_pct = st.number_input(
    "Current Deviation %",
    value=0.0
)

temperature_deviation_pct = st.number_input(
    "Temperature Deviation %",
    value=0.0
)

power_deviation_pct = st.number_input(
    "Power Deviation %",
    value=0.0
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Analyze Electrical Condition",
    use_container_width=True
):

    payload = {
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "power": power,

        "voltage_change": voltage_change,
        "current_change": current_change,
        "temperature_change": temperature_change,
        "power_change": power_change,

        "voltage_rolling_mean": voltage_rolling_mean,
        "current_rolling_mean": current_rolling_mean,
        "temperature_rolling_mean": temperature_rolling_mean,
        "power_rolling_mean": power_rolling_mean,

        "voltage_rolling_std": voltage_rolling_std,
        "current_rolling_std": current_rolling_std,
        "temperature_rolling_std": temperature_rolling_std,
        "power_rolling_std": power_rolling_std,

        "voltage_deviation": voltage_deviation,
        "current_deviation": current_deviation,
        "temperature_deviation": temperature_deviation,
        "power_deviation": power_deviation,

        "voltage_deviation_pct": voltage_deviation_pct,
        "current_deviation_pct": current_deviation_pct,
        "temperature_deviation_pct": temperature_deviation_pct,
        "power_deviation_pct": power_deviation_pct,
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            st.subheader("AI Prediction")

            if result["status"] == "ANOMALY":
                st.error("🚨 ANOMALY DETECTED")

            else:
                st.success("✅ SYSTEM NORMAL")

            st.metric(
                "Anomaly Probability",
                f"{result['anomaly_probability'] * 100:.1f}%"
            )

            st.write(
                f"**Prediction:** {result['prediction']}"
            )

            st.write(
                f"**Decision Threshold:** "
                f"{result['threshold'] * 100:.0f}%"
            )

            st.write(
                f"**Model:** {result['model']}"
            )

        else:

            st.error(
                f"API returned status code {response.status_code}"
            )

    except requests.exceptions.RequestException:

        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure Uvicorn is running."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Resilient-Grid | Random Forest V7 | "
    "Causal temporal feature engineering"
)
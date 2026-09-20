import time
import random

import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resilient-Grid Live Monitor",
    page_icon="⚡",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# PAGE TITLE
# ============================================================

st.title("⚡ Resilient-Grid Live Monitor")

st.write(
    "Simulated real-time electrical sensors monitored by "
    "Random Forest V7."
)


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "fault_mode" not in st.session_state:
    st.session_state.fault_mode = "Normal"


# ============================================================
# CONTROLS
# ============================================================

st.header("Simulation Control")

fault_mode = st.selectbox(
    "Sensor condition",
    [
        "Normal",
        "Overheating",
        "Combined Fault"
    ]
)

start = st.button(
    "▶ Start Live Monitoring",
    use_container_width=True
)

stop = st.button(
    "⏹ Stop Monitoring",
    use_container_width=True
)


if start:
    st.session_state.running = True

if stop:
    st.session_state.running = False


# ============================================================
# DISPLAY AREAS
# ============================================================

sensor_placeholder = st.empty()

status_placeholder = st.empty()

probability_placeholder = st.empty()

details_placeholder = st.empty()


# ============================================================
# SENSOR GENERATOR
# ============================================================

def generate_reading(mode):

    # --------------------------------------------------------
    # Normal readings
    # --------------------------------------------------------

    voltage = random.normalvariate(230, 2)
    current = random.normalvariate(10, 0.5)
    temperature = random.normalvariate(35, 1.5)

    # --------------------------------------------------------
    # Fault conditions
    # --------------------------------------------------------

    if mode == "Overheating":

        temperature = random.normalvariate(58, 2)

    elif mode == "Combined Fault":

        voltage = random.normalvariate(205, 3)
        current = random.normalvariate(18, 1)
        temperature = random.normalvariate(58, 2)

    power = voltage * current

    return {
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "power": power
    }


# ============================================================
# FEATURE GENERATION
# ============================================================

def create_features(reading):

    voltage = reading["voltage"]
    current = reading["current"]
    temperature = reading["temperature"]
    power = reading["power"]

    # --------------------------------------------------------
    # Simple simulated temporal changes
    # --------------------------------------------------------

    voltage_change = random.normalvariate(0, 1)
    current_change = random.normalvariate(0, 0.3)
    temperature_change = random.normalvariate(0, 0.5)
    power_change = random.normalvariate(0, 50)

    # --------------------------------------------------------
    # Simulated causal rolling baselines
    # --------------------------------------------------------

    voltage_rolling_mean = 230.0
    current_rolling_mean = 10.0
    temperature_rolling_mean = 35.0
    power_rolling_mean = 2300.0

    voltage_rolling_std = 2.0
    current_rolling_std = 1.0
    temperature_rolling_std = 2.0
    power_rolling_std = 100.0

    # --------------------------------------------------------
    # Fault-specific baseline adjustments
    # --------------------------------------------------------

    if st.session_state.fault_mode == "Overheating":

        temperature_change = 20.0
        temperature_rolling_mean = 37.0
        temperature_rolling_std = 4.0

    elif st.session_state.fault_mode == "Combined Fault":

        voltage_change = -10.0
        current_change = 8.0
        temperature_change = 20.0
        power_change = 1390.0

        voltage_rolling_mean = 228.0
        current_rolling_mean = 10.0
        temperature_rolling_mean = 37.0
        power_rolling_mean = 2280.0

        voltage_rolling_std = 5.0
        current_rolling_std = 2.0
        temperature_rolling_std = 4.0
        power_rolling_std = 250.0

    # --------------------------------------------------------
    # Deviations
    # --------------------------------------------------------

    voltage_deviation = (
        voltage - voltage_rolling_mean
    )

    current_deviation = (
        current - current_rolling_mean
    )

    temperature_deviation = (
        temperature - temperature_rolling_mean
    )

    power_deviation = (
        power - power_rolling_mean
    )

    # --------------------------------------------------------
    # Percentage deviations
    # --------------------------------------------------------

    voltage_deviation_pct = (
        voltage_deviation / voltage_rolling_mean
        if voltage_rolling_mean != 0
        else 0
    )

    current_deviation_pct = (
        current_deviation / current_rolling_mean
        if current_rolling_mean != 0
        else 0
    )

    temperature_deviation_pct = (
        temperature_deviation / temperature_rolling_mean
        if temperature_rolling_mean != 0
        else 0
    )

    power_deviation_pct = (
        power_deviation / power_rolling_mean
        if power_rolling_mean != 0
        else 0
    )

    return {
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


# ============================================================
# LIVE MONITORING LOOP
# ============================================================

if st.session_state.running:

    st.session_state.fault_mode = fault_mode

    while st.session_state.running:

        reading = generate_reading(
            st.session_state.fault_mode
        )

        payload = create_features(reading)

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=5
            )

            if response.status_code == 200:

                result = response.json()

                # --------------------------------------------
                # Sensor display
                # --------------------------------------------

                with sensor_placeholder.container():

                    st.subheader("Live Sensor Readings")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Voltage",
                        f"{reading['voltage']:.1f} V"
                    )

                    col2.metric(
                        "Current",
                        f"{reading['current']:.1f} A"
                    )

                    col1.metric(
                        "Temperature",
                        f"{reading['temperature']:.1f} °C"
                    )

                    col2.metric(
                        "Power",
                        f"{reading['power']:.0f} W"
                    )

                # --------------------------------------------
                # AI status
                # --------------------------------------------

                probability = result[
                    "anomaly_probability"
                ]

                if result["status"] == "ANOMALY":

                    status_placeholder.error(
                        "🚨 ANOMALY DETECTED"
                    )

                else:

                    status_placeholder.success(
                        "✅ SYSTEM NORMAL"
                    )

                probability_placeholder.metric(
                    "Anomaly Probability",
                    f"{probability * 100:.1f}%"
                )

                details_placeholder.write(
                    f"**Model:** {result['model']}  \n"
                    f"**Threshold:** "
                    f"{result['threshold'] * 100:.0f}%  \n"
                    f"**Prediction:** "
                    f"{result['prediction']}"
                )

            else:

                status_placeholder.error(
                    f"API Error: {response.status_code}"
                )

        except requests.exceptions.RequestException:

            status_placeholder.error(
                "FastAPI server is not running."
            )

        time.sleep(2)
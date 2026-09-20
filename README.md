# ⚡ Resilient-Grid

## Edge-Oriented AI for Electrical Anomaly Detection

Resilient-Grid is a machine-learning research prototype for detecting abnormal electrical operating conditions from simulated voltage, current, temperature, and power sensor data.

The project explores how temporal feature engineering and machine learning can be used to identify abnormal electrical behavior in a near-real-time monitoring scenario.

---

## 🎯 Project Objective

The objective of Resilient-Grid is to build an AI-based monitoring pipeline that can:

- simulate electrical sensor readings
- detect abnormal operating conditions
- engineer temporal features from previous readings
- compare multiple machine-learning approaches
- expose the trained model through an API
- visualize predictions through a web dashboard
- simulate live electrical monitoring
- analyze which features influence the model
- document model limitations and detection gaps

This is a research and portfolio prototype rather than a production electrical safety system.

---

# 🧠 System Architecture

```text
Simulated Electrical Sensors
          │
          ▼
Voltage / Current / Temperature / Power
          │
          ▼
Temporal Feature Engineering
          │
          ├── Changes
          ├── Rolling Means
          ├── Rolling Standard Deviations
          ├── Deviations
          └── Percentage Deviations
          │
          ▼
StandardScaler
          │
          ▼
Random Forest V7
          │
          ▼
Anomaly Probability
          │
          ▼
Decision Threshold
          │
          ├── NORMAL
          │
          └── ANOMALY
          │
          ▼
FastAPI
          │
          ├── Streamlit Dashboard
          │
          └── Live Monitor
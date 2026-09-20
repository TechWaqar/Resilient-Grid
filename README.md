# Resilient-Grid — Edge-AI Electrical Anomaly Detection

An end-to-end machine learning system for detecting abnormal electrical behavior and supporting intelligent monitoring of micro-grid environments.

The project simulates electrical sensor data, performs temporal feature engineering, trains and evaluates machine learning models, exposes predictions through a FastAPI service, and provides Streamlit monitoring dashboards.


## Project Objective

Electrical systems can develop abnormal behavior before a serious failure occurs.

The goal of Resilient-Grid is to investigate whether machine learning can identify abnormal electrical patterns from sensor measurements such as:

- Voltage
- Current
- Temperature
- Power

The project focuses on a complete machine learning workflow:

**Data → Feature Engineering → Model Training → Evaluation → API → Dashboard → Testing → Research Analysis**

---

## Core Idea

The system receives electrical measurements and transforms them into temporal features that describe both the current state and recent behavior of the system.

The final V7 model uses a Random Forest classifier with:

- Current electrical measurements
- Short-term changes
- Causal rolling statistics
- Deviations from recent operating conditions
- Percentage deviations

The temporal rolling features are calculated using **previous observations only**, making the feature-engineering approach more appropriate for online monitoring.

---

## System Architecture

```text
Simulated Electrical Sensors
            │
            ▼
   Voltage / Current /
 Temperature / Power
            │
            ▼
   Data Generation
            │
            ▼
 Temporal Feature Engineering
            │
            ▼
  Causal Rolling Features
            │
            ▼
 Random Forest Classifier
            │
            ▼
    Anomaly Probability
            │
       ┌────┴────┐
       ▼         ▼
   FastAPI    Streamlit
     API       Dashboard
       │         │
       └────┬────┘
            ▼
      Monitoring /
     Model Analysis

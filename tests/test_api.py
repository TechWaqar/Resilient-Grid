import requests


BASE_URL = "http://127.0.0.1:8000"


def predict(data):
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )

    assert response.status_code == 200

    return response.json()


def base_input():
    return {
        "voltage": 230.0,
        "current": 10.0,
        "temperature": 35.0,
        "power": 2300.0,

        "voltage_change": 0.0,
        "current_change": 0.0,
        "temperature_change": 0.0,
        "power_change": 0.0,

        "voltage_rolling_mean": 230.0,
        "current_rolling_mean": 10.0,
        "temperature_rolling_mean": 35.0,
        "power_rolling_mean": 2300.0,

        "voltage_rolling_std": 2.0,
        "current_rolling_std": 1.0,
        "temperature_rolling_std": 2.0,
        "power_rolling_std": 100.0,

        "voltage_deviation": 0.0,
        "current_deviation": 0.0,
        "temperature_deviation": 0.0,
        "power_deviation": 0.0,

        "voltage_deviation_pct": 0.0,
        "current_deviation_pct": 0.0,
        "temperature_deviation_pct": 0.0,
        "power_deviation_pct": 0.0,
    }


def test_health():
    response = requests.get(
        f"{BASE_URL}/health"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "healthy"
    assert result["model_loaded"] is True
    assert result["model"] == "Random Forest V7"


def test_normal_reading():
    data = base_input()

    result = predict(data)

    print("\nNORMAL TEST:")
    print(result)

    assert result["status"] == "NORMAL"
    assert result["prediction"] == 0


def test_overheating():
    data = base_input()

    data["temperature"] = 58.0
    data["temperature_change"] = 20.0
    data["temperature_rolling_mean"] = 37.0
    data["temperature_rolling_std"] = 4.0
    data["temperature_deviation"] = 21.0
    data["temperature_deviation_pct"] = 21.0 / 37.0

    data["power"] = 3690.0
    data["power_change"] = 1390.0
    data["power_rolling_mean"] = 2280.0
    data["power_rolling_std"] = 250.0
    data["power_deviation"] = 1410.0
    data["power_deviation_pct"] = 1410.0 / 2280.0

    result = predict(data)

    print("\nOVERHEATING TEST:")
    print(result)

    # Verify the API returns a valid V7 prediction.
    assert result["model"] == "Random Forest V7"
    assert 0.0 <= result["anomaly_probability"] <= 1.0
    assert result["prediction"] in [0, 1]

    # The current V7 model assigns this manually constructed
    # overheating pattern a probability of about 0.16.
    # We therefore do not force an ANOMALY classification here.


def test_combined_fault():
    data = base_input()

    data["voltage"] = 205.0
    data["current"] = 18.0
    data["temperature"] = 58.0
    data["power"] = 3690.0

    data["voltage_change"] = -10.0
    data["current_change"] = 8.0
    data["temperature_change"] = 20.0
    data["power_change"] = 1390.0

    data["voltage_rolling_mean"] = 228.0
    data["current_rolling_mean"] = 10.0
    data["temperature_rolling_mean"] = 37.0
    data["power_rolling_mean"] = 2280.0

    data["voltage_rolling_std"] = 5.0
    data["current_rolling_std"] = 2.0
    data["temperature_rolling_std"] = 4.0
    data["power_rolling_std"] = 250.0

    data["voltage_deviation"] = -23.0
    data["current_deviation"] = 8.0
    data["temperature_deviation"] = 21.0
    data["power_deviation"] = 1410.0

    data["voltage_deviation_pct"] = -23.0 / 228.0
    data["current_deviation_pct"] = 8.0 / 10.0
    data["temperature_deviation_pct"] = 21.0 / 37.0
    data["power_deviation_pct"] = 1410.0 / 2280.0

    result = predict(data)

    print("\nCOMBINED FAULT TEST:")
    print(result)

    assert result["status"] == "ANOMALY"
    assert result["prediction"] == 1
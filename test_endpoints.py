import requests

BASE_URL = "http://localhost:5000"

def test_endpoints():
    print("Testing /detect-appliances...")
    res = requests.post(f"{BASE_URL}/detect-appliances", json={"mains_sequence": [120, 140, 200, 600, 750, 820, 800, 790, 800, 750]})
    print(res.status_code, res.json())
    print("\n----------------\n")

    print("Testing /predict-appliance-power...")
    res = requests.post(f"{BASE_URL}/predict-appliance-power", json={"mains_sequence": [120, 140, 200, 600, 750, 820, 800, 790, 800, 750]})
    print(res.status_code, res.json())
    print("\n----------------\n")

    print("Testing /predict-cost...")
    res = requests.post(f"{BASE_URL}/predict-cost", json={"daily_usage_kwh": 8.2})
    print(res.status_code, res.json())
    print("\n----------------\n")

    print("Testing /predict-month-end...")
    res = requests.post(f"{BASE_URL}/predict-month-end", json={"current_units": 120, "days_elapsed": 10})
    print(res.status_code, res.json())
    print("\n----------------\n")

    print("Testing /detect-anomaly...")
    res = requests.post(f"{BASE_URL}/detect-anomaly", json={"mains_sequence": [100, 105, 95, 110, 100, 2500, 105, 90, 115, 100]})
    print(res.status_code, res.json())

if __name__ == "__main__":
    test_endpoints()

"""
Transaction simulator for the Fraud Detection API.

Sends realistic fake transactions at a 2-second interval.
Set API_URL env var to target a remote deployment:
    API_URL=https://your-app.onrender.com python scripts/simulate_transactions.py
"""

import os
import sys
import time
import random
import json
from datetime import datetime

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_URL = f"{API_URL}/predict"


def _normal_v_features() -> dict:
    """Return V1-V28 drawn from a standard normal distribution (typical for legit transactions)."""
    return {f"v{i}": round(random.gauss(0, 1), 6) for i in range(1, 29)}


def _anomalous_v_features() -> dict:
    """Return V1-V28 with exaggerated values that resemble fraud patterns."""
    features = {f"v{i}": round(random.gauss(0, 1), 6) for i in range(1, 29)}
    # Key discriminative features identified by model feature importance
    features["v1"] = round(random.uniform(-10, -5), 6)
    features["v3"] = round(random.uniform(-12, -6), 6)
    features["v4"] = round(random.uniform(4, 8), 6)
    features["v10"] = round(random.uniform(-8, -4), 6)
    features["v14"] = round(random.uniform(-12, -7), 6)
    return features


def generate_legitimate_transaction() -> dict:
    tx = {
        "amount": round(random.uniform(5, 200), 2),
        "hour_of_day": round(random.uniform(8, 22), 2),
        "day_of_week": random.randint(0, 6),
    }
    tx.update(_normal_v_features())
    return tx


def generate_fraud_transaction() -> dict:
    tx = {
        "amount": round(random.uniform(500, 5000), 2),
        "hour_of_day": round(random.uniform(1, 5), 2),
        "day_of_week": random.randint(0, 6),
    }
    tx.update(_anomalous_v_features())
    return tx


def format_hour(hour: float) -> str:
    h = int(hour)
    m = int((hour - h) * 60)
    period = "am" if h < 12 else "pm"
    display_h = h if h <= 12 else h - 12
    display_h = 12 if display_h == 0 else display_h
    return f"{display_h}:{m:02d}{period}"


def main():
    print(f"\nFraud Detection Transaction Simulator")
    print(f"Target API: {API_URL}")
    print(f"Press Ctrl+C to stop\n")
    print(f"{'#':<6} {'Amount':<12} {'Time':<10} {'Result':<20} {'Confidence':<14} {'Latency'}")
    print("-" * 80)

    total = 0
    fraud_count = 0
    error_count = 0
    total_latency = 0.0

    try:
        with httpx.Client(timeout=10.0) as client:
            while True:
                total += 1
                is_fraud_sim = random.random() < 0.05
                tx = generate_fraud_transaction() if is_fraud_sim else generate_legitimate_transaction()

                try:
                    t0 = time.perf_counter()
                    response = client.post(PREDICT_URL, json=tx)
                    api_latency = (time.perf_counter() - t0) * 1000
                    response.raise_for_status()
                    data = response.json()

                    detected_fraud = data["is_fraud"]
                    confidence = data["confidence"]
                    inference_ms = data["inference_time_ms"]
                    risk = data["risk_level"]
                    total_latency += api_latency

                    if detected_fraud:
                        fraud_count += 1
                        result_str = "\U0001f6a8 FRAUD    "
                    else:
                        result_str = "✅ LEGITIMATE"

                    hour_str = format_hour(tx["hour_of_day"])
                    amount_str = f"${tx['amount']:,.2f}"

                    print(
                        f"#{total:<5} {amount_str:<12} {hour_str:<10} "
                        f"{result_str:<20} confidence: {confidence:.2f}  "
                        f"{inference_ms:.1f}ms  [{risk}]"
                    )

                except httpx.ConnectError:
                    error_count += 1
                    print(f"#{total:<5} ERROR: Cannot connect to {API_URL}. Is the API running?")
                except httpx.HTTPStatusError as exc:
                    error_count += 1
                    print(f"#{total:<5} ERROR: HTTP {exc.response.status_code} — {exc.response.text[:60]}")
                except Exception as exc:
                    error_count += 1
                    print(f"#{total:<5} ERROR: {exc}")

                time.sleep(2)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)
        print(f"  Total transactions:    {total}")
        print(f"  Fraud detected:        {fraud_count} ({fraud_count/max(total,1)*100:.1f}%)")
        print(f"  Errors:                {error_count}")
        if total - error_count > 0:
            avg_latency = total_latency / (total - error_count)
            print(f"  Avg API latency:       {avg_latency:.1f}ms")
        print("Simulation stopped.")


if __name__ == "__main__":
    main()

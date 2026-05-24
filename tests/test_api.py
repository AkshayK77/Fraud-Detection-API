"""
API test suite for the Fraud Detection API.

Run with:
    pytest tests/test_api.py -v
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app

LEGITIMATE_TRANSACTION = {
    "amount": 45.20,
    "hour_of_day": 14.0,
    "day_of_week": 2,
    "v1": -0.5,
    "v2": 0.3,
    "v3": 1.2,
    "v4": 0.8,
    "v5": -0.2,
    "v6": 0.4,
    "v7": 0.1,
    "v8": 0.05,
    "v9": 0.2,
    "v10": 0.1,
    "v11": -0.3,
    "v12": -0.4,
    "v13": -0.5,
    "v14": -0.1,
    "v15": 0.6,
    "v16": -0.2,
    "v17": 0.1,
    "v18": 0.01,
    "v19": 0.2,
    "v20": 0.1,
    "v21": -0.01,
    "v22": 0.1,
    "v23": -0.05,
    "v24": 0.03,
    "v25": 0.06,
    "v26": -0.09,
    "v27": 0.06,
    "v28": -0.01,
}

FRAUD_TRANSACTION = {
    "amount": 2847.50,
    "hour_of_day": 2.5,
    "day_of_week": 6,
    "v1": -8.5,
    "v2": 3.2,
    "v3": -9.8,
    "v4": 6.1,
    "v5": -5.4,
    "v6": -3.2,
    "v7": -7.1,
    "v8": 2.1,
    "v9": -4.3,
    "v10": -7.2,
    "v11": 3.5,
    "v12": -8.9,
    "v13": 2.3,
    "v14": -10.4,
    "v15": 0.5,
    "v16": -5.1,
    "v17": -6.3,
    "v18": 2.8,
    "v19": 0.4,
    "v20": 0.3,
    "v21": 0.6,
    "v22": 0.5,
    "v23": -0.3,
    "v24": 0.2,
    "v25": 0.1,
    "v26": -0.2,
    "v27": 0.3,
    "v28": 0.1,
}


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_legitimate_transaction(client: AsyncClient):
    response = await client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
    assert data["risk_level"] == "LOW"
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_fraud_transaction(client: AsyncClient):
    response = await client.post("/predict", json=FRAUD_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["risk_level"] in ("MEDIUM", "HIGH")
    assert data["confidence"] >= 0.5


@pytest.mark.asyncio
async def test_invalid_input_missing_fields(client: AsyncClient):
    response = await client.post("/predict", json={"amount": 100.0})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_input_bad_hour(client: AsyncClient):
    bad_tx = {**LEGITIMATE_TRANSACTION, "hour_of_day": 25.0}
    response = await client.post("/predict", json=bad_tx)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_inference_speed(client: AsyncClient):
    import time

    for i in range(10):
        t0 = time.perf_counter()
        response = await client.post("/predict", json=LEGITIMATE_TRANSACTION)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert response.status_code == 200
        data = response.json()
        assert data["inference_time_ms"] < 100, (
            f"Request {i+1} inference took {data['inference_time_ms']:.1f}ms (>100ms)"
        )


@pytest.mark.asyncio
async def test_prediction_response_schema(client: AsyncClient):
    response = await client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()

    required_fields = {"is_fraud", "confidence", "risk_level", "inference_time_ms"}
    assert required_fields.issubset(data.keys()), (
        f"Missing fields: {required_fields - data.keys()}"
    )

    assert isinstance(data["is_fraud"], bool)
    assert isinstance(data["confidence"], float)
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(data["inference_time_ms"], float)
    assert 0.0 <= data["confidence"] <= 1.0

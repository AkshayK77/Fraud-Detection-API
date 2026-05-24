# Fraud Detection API

A production-ready REST API that classifies credit card transactions as fraudulent or legitimate in real time using a Random Forest model trained on 284,807 real-world transactions.

## Architecture

```
[Transaction] → [FastAPI] → [Random Forest Model] → [Prediction]
                    ↓
              [GitHub Actions CI/CD]
                    ↓
              [Render Deployment]
```

Incoming transactions are validated by Pydantic, passed to a singleton `ModelHandler` that scales the amount feature and runs inference on a pre-trained Random Forest classifier. The model is loaded once at startup and shared across all requests. GitHub Actions runs the full test suite on every push; passing builds auto-deploy to Render via Docker.

## ML Model

| Property | Value |
|---|---|
| Dataset | 284,807 transactions, 0.17% fraud rate |
| Class balancing | SMOTE (synthetic minority oversampling) |
| Algorithm | Random Forest (100 estimators, balanced class weight) |
| Features | V1–V28 (PCA), Amount (scaled), hour_of_day, day_of_week |
| ROC-AUC | 0.9609 |
| Precision (fraud) | 0.77 |
| Recall (fraud) | 0.82 |
| F1-score (fraud) | 0.79 |
| True Positives | 80 / 98 fraud cases caught |
| False Positives | 24 legitimate transactions flagged |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Welcome message and links |
| GET | `/health` | Liveness check — returns model status and version |
| POST | `/predict` | Classify a transaction, returns fraud flag + confidence + risk level |
| GET | `/docs` | Swagger UI (interactive documentation) |
| GET | `/redoc` | ReDoc documentation |

### POST /predict — Request Body

```json
{
  "amount": 149.62,
  "hour_of_day": 14.5,
  "day_of_week": 2,
  "v1": -1.36,
  "v2": -0.07,
  ...
  "v28": -0.02
}
```

### POST /predict — Response

```json
{
  "is_fraud": false,
  "confidence": 0.0312,
  "risk_level": "LOW",
  "inference_time_ms": 14.7
}
```

`risk_level` thresholds: `HIGH` ≥ 0.8 · `MEDIUM` ≥ 0.5 · `LOW` < 0.5

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/fraud-detection-api.git
cd fraud-detection-api

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Place creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# into the data/ directory

# 5. Train the model (run all cells)
jupyter notebook notebooks/train_model.ipynb

# 6. Start the API
uvicorn app.main:app --reload

# 7. Open the interactive docs
# http://localhost:8000/docs
```

## Running Tests

```bash
pytest tests/ -v
```

## Demo — Transaction Simulator

The simulator fires one transaction every 2 seconds and prints live results:

```bash
python scripts/simulate_transactions.py
```

Example output:
```
#1     $45.20       2:15pm    ✅ LEGITIMATE        confidence: 0.04  12.1ms  [LOW]
#2     $1,240.50    3:17am    🚨 FRAUD             confidence: 0.94  18.3ms  [HIGH]
#3     $89.99       11:30am   ✅ LEGITIMATE        confidence: 0.02  11.8ms  [LOW]
```

To target the deployed API:
```bash
API_URL=https://fraud-detection-api.onrender.com python scripts/simulate_transactions.py
```

## Deployment

### CI/CD (GitHub Actions)

On every push to `main`:
1. Python 3.11 environment is set up
2. Dependencies are installed
3. A dummy model is created so tests run without the full dataset
4. `pytest` runs the full test suite
5. Passing builds trigger auto-deploy on Render

### Render (Docker)

1. Create a new account at [render.com](https://render.com)
2. Click **New → Web Service** → connect your GitHub repo
3. Render detects the `Dockerfile` automatically
4. Set **Health Check Path** to `/health`
5. Click **Deploy** — your API will be live at `https://<your-service>.onrender.com`

The `render.yaml` in this repo enables one-click deploy via the Render Blueprint.

> **Note:** The free Render tier spins down after 15 minutes of inactivity. The first request after a cold start may take 30–60 seconds.

## Tech Stack

- **FastAPI** — high-performance async REST framework
- **Scikit-learn** — Random Forest classifier
- **Imbalanced-learn** — SMOTE oversampling
- **Pandas / NumPy** — data processing and feature engineering
- **Joblib** — model serialisation
- **Pydantic v2** — request / response validation
- **Uvicorn** — ASGI server
- **Pytest + httpx** — async test suite
- **Docker** — containerisation
- **GitHub Actions** — CI/CD pipeline
- **Render** — cloud deployment

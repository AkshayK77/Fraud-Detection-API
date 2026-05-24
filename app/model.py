import time
import logging
import numpy as np
import joblib
from pathlib import Path
from app.schemas import TransactionInput, PredictionResponse

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "model" / "fraud_model.pkl"
SCALER_PATH = Path(__file__).parent.parent / "model" / "scaler.pkl"

# Feature order must exactly match the training notebook
FEATURE_ORDER = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount", "hour_of_day", "day_of_week",
]


class ModelHandler:
    _instance: "ModelHandler | None" = None

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.is_loaded = True
        logger.info("Model and scaler loaded from %s", MODEL_PATH.parent)

    @classmethod
    def get_instance(cls) -> "ModelHandler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, transaction: TransactionInput) -> PredictionResponse:
        start = time.perf_counter()

        amount_scaled = float(self.scaler.transform([[transaction.amount]])[0][0])

        features = np.array([[
            transaction.v1, transaction.v2, transaction.v3, transaction.v4,
            transaction.v5, transaction.v6, transaction.v7, transaction.v8,
            transaction.v9, transaction.v10, transaction.v11, transaction.v12,
            transaction.v13, transaction.v14, transaction.v15, transaction.v16,
            transaction.v17, transaction.v18, transaction.v19, transaction.v20,
            transaction.v21, transaction.v22, transaction.v23, transaction.v24,
            transaction.v25, transaction.v26, transaction.v27, transaction.v28,
            amount_scaled,
            transaction.hour_of_day,
            transaction.day_of_week,
        ]], dtype=np.float64)

        fraud_probability = float(self.model.predict_proba(features)[0][1])
        is_fraud = bool(self.model.predict(features)[0] == 1)

        inference_time_ms = (time.perf_counter() - start) * 1000

        if fraud_probability >= 0.8:
            risk_level = "HIGH"
        elif fraud_probability >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            is_fraud=is_fraud,
            confidence=round(fraud_probability, 4),
            risk_level=risk_level,
            inference_time_ms=round(inference_time_ms, 3),
        )

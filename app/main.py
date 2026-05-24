import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import TransactionInput, PredictionResponse
from app.model import ModelHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Fraud Detection API v%s", VERSION)
    try:
        ModelHandler.get_instance()
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
    yield
    logger.info("Shutting down Fraud Detection API")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection using a Random Forest model "
        "trained on 284,807 transactions with SMOTE class balancing."
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Fraud Detection API is running",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
async def health_check():
    handler = ModelHandler.get_instance()
    return {
        "status": "ok",
        "model_loaded": handler.is_loaded,
        "version": VERSION,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    try:
        handler = ModelHandler.get_instance()
        result = handler.predict(transaction)

        logger.info(
            "Prediction | amount=$%.2f | is_fraud=%s | confidence=%.4f | "
            "risk=%s | inference=%.2fms",
            transaction.amount,
            result.is_fraud,
            result.confidence,
            result.risk_level,
            result.inference_time_ms,
        )

        return result

    except ValueError as exc:
        logger.warning("Invalid input: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

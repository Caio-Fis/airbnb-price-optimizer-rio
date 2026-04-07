"""
FastAPI: API de predição de preço Airbnb.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from src.serving.predict import Predictor
from src.serving.schemas import HealthResponse, PredictionRequest, PredictionResponse

predictor: Predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    predictor = Predictor()
    if predictor.is_ready():
        logger.info("Model ready.")
    else:
        logger.warning("Model not loaded — predictions will fail until model is available.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Airbnb Price Optimizer API",
    description="Predição de preço ótimo para acomodações Airbnb usando ML + análise de imagens.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if predictor and predictor.is_ready() else "degraded",
        model_loaded=predictor is not None and predictor.is_ready(),
        model_version=predictor.model_version if predictor else "none",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not predictor or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predictor.predict(request)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Airbnb Price Optimizer API", "docs": "/docs"}

"""
FastAPI — Airbnb Price Optimizer API.

Segurança implementada:
  - Autenticação por API Key (X-API-Key header)
  - Rate limiting por IP (slowapi)
  - Security headers (nosniff, frame deny, XSS, referrer)
  - CORS configurável via env (não wildcard em produção)
  - Docs desabilitados em ENVIRONMENT=production
  - Erros internos nunca expostos ao cliente
  - Validação de domínio: bbox Rio, formato de bairro, room_type
  - Logging estruturado por request (sem corpo — sem PII)
"""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.serving.predict import Predictor
from src.serving.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.serving.security import (
    limiter,
    require_api_key,
    validate_coordinates,
    validate_neighbourhood,
    validate_room_type,
)
from src.serving.settings import settings

# ── Lifespan ──────────────────────────────────────────────────────────────────
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model and artefacts...")
    predictor = Predictor()
    if predictor.is_ready():
        logger.info("Model ready.")
    else:
        logger.warning("Model not loaded — /predict will return 503 until model is available.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Airbnb Price Optimizer API",
    description=(
        "Previsão de preço ótimo para acomodações Airbnb no Rio de Janeiro.\n\n"
        "**Autenticação**: envie o header `X-API-Key` com sua chave de acesso."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded. Try again later."},
        headers={"Retry-After": "60"},
    )


# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
    allow_credentials=False,
)


# ── Security headers middleware ───────────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    # Em produção com HTTPS, adicionar HSTS no proxy reverso (nginx/traefik)
    return response


# ── Request logging middleware (sem body — sem PII) ───────────────────────────
@app.middleware("http")
async def request_logger(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed_ms:.1f}ms) "
        f"ip={request.client.host if request.client else 'unknown'}"
    )
    response.headers["X-Request-Id"] = request_id
    return response


# ── Global exception handler (nunca expõe stack trace) ───────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again later."},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health(request: Request):
    """Healthcheck público — usado por load balancers e orquestradores."""
    return HealthResponse(
        status="ok" if predictor and predictor.is_ready() else "degraded",
        model_loaded=predictor is not None and predictor.is_ready(),
        model_version=predictor.model_version if predictor else "none",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Previsão de preço ótimo",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit_predict)
async def predict(request: Request, response: Response, body: PredictionRequest):
    """
    Retorna:
    - **predicted_price**: benchmark de mercado (o que listings similares cobram)
    - **revenue_optimal_price**: preço que maximiza receita esperada
    - **expected_occupancy_pct**: ocupação estimada no preço ótimo
    - **pricing_strategy**: `revenue_optimal` | `premium_positioning` | `fallback`
    """
    if not predictor or not predictor.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model not available")

    # Validações de domínio
    validate_neighbourhood(body.neighbourhood)
    validate_room_type(body.room_type)
    validate_coordinates(body.latitude, body.longitude)

    try:
        return predictor.predict(body)
    except Exception as e:
        logger.error(f"Prediction failed for neighbourhood={body.neighbourhood}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again later.",
        )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Airbnb Price Optimizer API",
        "version": "1.0.0",
        "docs": settings.docs_url or "disabled in production",
    }

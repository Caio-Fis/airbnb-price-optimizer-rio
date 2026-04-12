FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

# Artefatos de inferência (modelo + encoders + lookup tables)
COPY models/model.joblib ./models/model.joblib
COPY models/encoders.joblib ./models/encoders.joblib
COPY data/processed/feature_names.joblib ./data/processed/feature_names.joblib
COPY data/processed/competition_stats.joblib ./data/processed/competition_stats.joblib
COPY data/processed/demand_params.joblib ./data/processed/demand_params.joblib
COPY data/processed/hw_seasonal_by_dow.joblib ./data/processed/hw_seasonal_by_dow.joblib
COPY data/processed/geo_metro_cache.json ./data/processed/geo_metro_cache.json
COPY data/processed/prediction_intervals.joblib ./data/processed/prediction_intervals.joblib
COPY data/processed/listings_slim.parquet ./data/processed/listings_slim.parquet

ENV MODEL_PATH=models/model.joblib \
    ENCODERS_PATH=models/encoders.joblib \
    PROCESSED_DATA_PATH=data/processed \
    ENVIRONMENT=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]

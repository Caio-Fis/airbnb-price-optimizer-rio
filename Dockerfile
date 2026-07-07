# Imagem única do app (Hugging Face Spaces / local):
# stage 1 builda o frontend React, stage 2 roda a FastAPI servindo API + estáticos.

# ── Stage 1: build do frontend (Vite/React) ───────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: API + artefatos ──────────────────────────────────────────────────
# 3.12: todas as deps pinadas têm wheels; artefatos joblib carregam entre versões
# de Python desde que as versões das libs (sklearn/lightgbm) sejam as mesmas
FROM python:3.12-slim
WORKDIR /app

# libgomp: exigida por LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY models/ models/
COPY data/external/ data/external/
COPY data/processed/feature_names.joblib \
     data/processed/competition_stats.joblib \
     data/processed/demand_params.joblib \
     data/processed/hw_seasonal_by_dow.joblib \
     data/processed/prediction_intervals.joblib \
     data/processed/seasonal_factors.joblib \
     data/processed/geo_metro_cache.json \
     data/processed/geo_poi_bar.json \
     data/processed/geo_poi_restaurant.json \
     data/processed/geo_poi_cafe_bakery.json \
     data/processed/geo_poi_gym.json \
     data/processed/geo_poi_park.json \
     data/processed/geo_poi_supermarket.json \
     data/processed/geo_poi_attraction.json \
     data/processed/geo_poi_nightclub.json \
     data/processed/clip_features.parquet \
     data/processed/listings_slim.parquet \
     data/processed/listings_map.json.gz \
     data/processed/

COPY --from=frontend /build/dist frontend/dist

# HF Spaces exige a porta 7860; usuário não-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 7860

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "7860"]

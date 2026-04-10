# Airbnb Price Optimizer — Rio de Janeiro

Pipeline de ML de ponta a ponta para **otimização de preço** de listings do Airbnb no Rio de Janeiro. O sistema encontra o `revenue_optimal_price` — o preço que maximiza receita esperada (preço × ocupação) — usando uma curva de demanda estimada a partir de dois snapshots do Inside Airbnb (Jun/2025 e Set/2025).

**API pública:** `https://airbnb-price-api-966533570956.us-central1.run.app`

**Monitoramento:** <a href="https://storage.googleapis.com/airbnb-rj-projectb0f/evidently/monitoring_drift_report.html" target="_blank">Relatório de Data Drift Evidently — Jun vs Set 2025</a>

---

## Objetivo

Dado um listing do Airbnb (bairro, tipo de quarto, comodidades, localização etc.), a API retorna:

| Campo | Descrição |
|---|---|
| `predicted_price` | Benchmark de mercado — o que listings similares cobram |
| `revenue_optimal_price` | Preço que maximiza receita esperada (preço × ocupação) |
| `expected_occupancy_pct` | Ocupação estimada (%) no preço ótimo |
| `pricing_strategy` | `revenue_optimal` / `premium_positioning` / `fallback` |
| `confidence` | `low` / `medium` / `high` baseado em histórico de reviews |
| `local_median_price` | P50 do segmento (bairro × tipo de quarto) |
| `price_range_low/high` | Intervalo de confiança data-driven (P10/P90 dos resíduos OOF) |

---

## Estimação de Demanda

A inovação central do projeto: usar os **dois snapshots** (Jun/2025 → Set/2025) para estimar elasticidade-preço por segmento (bairro × tipo de quarto).

Para cada listing presente nos dois períodos, estimamos:

```
Δlogit(occupancy) ~ b × Δlog(price)
```

O coeficiente `b` é a elasticidade. Com ela, calculamos o preço ótimo:

| Estratégia | Condição | Como funciona |
|---|---|---|
| `revenue_optimal` | `b < -1` (elástica) | Grid search: `argmax[ price × sigmoid(a + b·log(price/p50)) ]` |
| `premium_positioning` | `-1 ≤ b < 0` (inelástica) | Sobe até P75 local — perde pouca ocupação, ganha margem |
| `fallback` | Dados insuficientes | Retorna mediana local |

256 segmentos com parâmetros estimados, cobrindo os principais bairros do Rio.

---

## Modelo

- **Algoritmo:** LightGBM (vencedor) e XGBoost, comparados via 5-fold CV
- **Target:** `log(price)` — previsão em escala log, convertida com `expm1`
- **Métricas OOF** (após winsorização P1–P99 do target):

| Modelo | RMSE (R$) | MAE (R$) |
|---|---|---|
| LightGBM | **104.56** | **20.50** |
| XGBoost | 109.33 | 20.53 |

- **89 features**: tabular, amenities (MLB), bairro (target encoding), geo (Haversine), reviews, competição local, sazonalidade (Holt-Winters), **demand features** (occupancy, price premium, revenue optimal price)
- **Tracking:** MLflow com registro de runs, métricas por fold e artefatos
- **Intervalo de confiança** data-driven: P10/P90 calculados sobre resíduos OOF (substitui ±15% fixo)

---

## Arquitetura

```
Inside Airbnb (Jun + Set 2025)
        │
        ▼
┌─────────────────┐
│  Ingestion DAG  │  download_data.py → data/raw/
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  tabular, geo, reviews, competition, demand
│       DAG           │  → data/processed/final_features.parquet
└────────┬────────────┘      demand_params.joblib (curva de demanda)
         │
         ▼
┌──────────────┐
│ Training DAG │  XGBoost / LightGBM (5-fold CV, target: log price)
│              │  → MLflow tracking → models/model.joblib
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────────┐
│  FastAPI  (Cloud Run — us-central1)        │
│  POST /predict                             │
│    → _build_features()                     │
│    → model.predict() → predicted_price     │
│    → _revenue_optimal() → optimal_price    │
└────────────────────────────────────────────┘
```

---

## Stack

| Camada | Tecnologia |
|---|---|
| Orquestração | Apache Airflow (Docker Compose) |
| Feature Engineering | pandas, statsmodels (Holt-Winters), scikit-learn |
| Modelos | XGBoost 2.0, LightGBM 4.3 |
| Tracking | MLflow |
| Monitoramento | Evidently AI (data drift) |
| Serving | FastAPI + uvicorn + slowapi |
| Deploy | Docker → Artifact Registry → Cloud Run (GCP) |
| Dados | DVC + GCS |

---

## Como usar a API

```bash
curl -X POST https://airbnb-price-api-966533570956.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "neighbourhood": "Copacabana",
    "room_type": "Entire home/apt",
    "accommodates": 4,
    "bathrooms": 1.0,
    "bedrooms": 2,
    "beds": 2,
    "minimum_nights": 2,
    "maximum_nights": 30,
    "number_of_reviews": 15,
    "amenities": ["Wifi", "Kitchen", "Air conditioning"],
    "host_is_superhost": false,
    "instant_bookable": true,
    "calculated_host_listings_count": 1,
    "availability_365": 200,
    "latitude": -22.9711,
    "longitude": -43.1823,
    "target_date": "2025-12-20"
  }'
```

**Resposta:**
```json
{
  "predicted_price": 339.16,
  "price_range_low": 288.29,
  "price_range_high": 390.04,
  "confidence": "medium",
  "local_median_price": 323.0,
  "seasonal_note": "sábado, mês 12",
  "revenue_optimal_price": 507.0,
  "expected_occupancy_pct": 40.8,
  "pricing_strategy": "premium_positioning"
}
```

### Campos do Request

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `neighbourhood` | string | sim | Bairro do listing |
| `room_type` | string | sim | `Entire home/apt`, `Private room`, `Shared room`, `Hotel room` |
| `accommodates` | int | sim | Número de hóspedes |
| `bathrooms` | float | sim | Número de banheiros |
| `bedrooms` | int | sim | Número de quartos |
| `beds` | int | sim | Número de camas |
| `minimum_nights` | int | sim | Mínimo de noites |
| `maximum_nights` | int | sim | Máximo de noites |
| `number_of_reviews` | int | sim | Total de reviews |
| `amenities` | list[string] | sim | Lista de comodidades |
| `host_is_superhost` | bool | sim | Superhost? |
| `instant_bookable` | bool | sim | Reserva instantânea? |
| `calculated_host_listings_count` | int | sim | Total de listings do host |
| `availability_365` | int | sim | Dias disponíveis no ano |
| `latitude` | float | não | Latitude (melhora features geo) |
| `longitude` | float | não | Longitude (melhora features geo) |
| `target_date` | string | não | Data alvo `YYYY-MM-DD` (melhora sazonalidade) |

---

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_eda.ipynb` | EDA de listings e calendário — distribuições, sazonalidade, comparação Jun/Set |
| `02_feature_engineering.ipynb` | Validação de features: HW, target encoding, reviews, geo, estimação de demanda e curvas de elasticidade |
| `02_reviews_analysis.ipynb` | Análise de sentimento, velocidade/recência, BERTopic |
| `03_modeling.ipynb` | XGBoost vs LightGBM, SHAP, análise de erros, pipeline de otimização, documentação de métricas |

---

## Dados

Fonte: [Inside Airbnb](http://insideairbnb.com/) — Rio de Janeiro

| Arquivo | Descrição |
|---|---|
| `listings_jun2025.parquet` | 38.4k listings, Jun/2025 |
| `listings_set2025.parquet` | 38.1k listings, Set/2025 (base de treino) |
| `calendar_jun2025.parquet` | 15.5M linhas — disponibilidade diária |
| `calendar_set2025.parquet` | 15.7M linhas — disponibilidade diária |
| `reviews_set2025.parquet` | 1.1M reviews com texto |

> Os dados não estão versionados no git. Use `dvc pull` para baixar do GCS.

---

## Como rodar localmente

### API sem Docker
```bash
pip install -r requirements-api.txt

MODEL_PATH=models/model.joblib \
ENCODERS_PATH=models/encoders.joblib \
PROCESSED_DATA_PATH=data/processed \
ENVIRONMENT=development \
uvicorn src.serving.main:app --port 8000
```

### Stack completa (Airflow + MLflow + API)
```bash
cp .env.example .env  # preencha as variáveis
docker compose up
```

Serviços:
- Airflow: http://localhost:8080 (admin/admin)
- MLflow: http://localhost:5000
- API: http://localhost:8000

---

## Deploy (Cloud Run)

```bash
docker build -f docker/api/Dockerfile -t airbnb-api:latest .
docker tag airbnb-api:latest us-central1-docker.pkg.dev/YOUR_PROJECT_ID/airbnb-optimizer/api:latest
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/airbnb-optimizer/api:latest

gcloud run deploy airbnb-price-api \
  --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/airbnb-optimizer/api:latest \
  --region=us-central1 \
  --allow-unauthenticated \
  --memory=2Gi --cpu=1 \
  --min-instances=0 --max-instances=3 \
  --port=8000
```

---

## Estrutura do Projeto

```
├── dags/                    # Airflow DAGs (ingestion, features, training, monitoring)
├── data/                    # Dados (não versionados — DVC)
│   ├── raw/                 # CSV.gz do Inside Airbnb
│   └── processed/           # Parquet + artefatos de inferência
├── docker/                  # Dockerfiles (api, airflow, mlflow)
├── docs/evidence/           # Relatório Evidently de drift
├── infra/                   # cloudrun.yaml, cloudbuild.yaml
├── models/                  # Modelo e encoders de produção
├── notebooks/               # EDA, feature engineering, modeling
├── scripts/                 # convert_to_parquet.py
├── src/
│   ├── features/            # Pipelines: tabular, geo, reviews, competition, demand
│   ├── ingestion/           # Download Inside Airbnb
│   ├── monitoring/          # Evidently drift detection
│   ├── serving/             # FastAPI (main, predict, schemas, security, settings)
│   └── training/            # Treino XGB/LGB + MLflow
├── tests/                   # 17 testes (api, features, model)
├── requirements.txt         # Dependências completas
├── requirements-api.txt     # Dependências mínimas (serving)
└── docker-compose.yml       # Stack local completa
```

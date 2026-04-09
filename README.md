# Airbnb Price Optimizer — Rio de Janeiro

Pipeline de ML de ponta a ponta para **otimização de preço** de listings do Airbnb no Rio de Janeiro. O sistema encontra o `revenue_optimal_price` — o preço que maximiza receita esperada (preço × ocupação) — usando uma curva de demanda estimada a partir de dois snapshots do Inside Airbnb (Jun/2025 e Set/2025).

**API pública:** `https://airbnb-price-api-966533570956.us-central1.run.app`

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
| `price_range_low/high` | Intervalo ±15% em torno do `predicted_price` |

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
| Tracking | MLflow (local) |
| Serving | FastAPI + uvicorn + slowapi |
| Deploy | Docker → Artifact Registry → Cloud Run (GCP) |
| Dados | DVC + GCS |

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

## Como rodar localmente

### Pré-requisitos
- Python 3.12+
- Docker

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

## Pipeline de Features

```
listings_set2025.parquet
        │
        ├── TabularFeaturePipeline    → preço, amenities, room_type, bairro
        ├── SeasonalityPipeline       → Holt-Winters (7-day), hw_seasonal por DOW
        ├── GeoFeaturePipeline        → distâncias Haversine a 9 POIs + metrô (OSM)
        ├── ReviewFeaturePipeline     → velocity, recência, keyword negativo
        ├── CompetitionPipeline       → P25/P50/P75/rank por (bairro × room_type)
        └── DemandPipeline            → elasticidade painel Jun→Set, revenue_optimal
                │
                ▼
        final_features.parquet  (~85 features, 38k listings)
```

### Estimação de Demanda

Para cada segmento (bairro × tipo de quarto), usamos listings presentes nos dois snapshots para estimar:

```
Δlogit(occupancy) ~ b × Δlog(price)
```

- `b < -1`: demanda elástica → grid search maximiza `price × occupancy`
- `-1 ≤ b < 0`: demanda inelástica → `premium_positioning` (sobe até P75)
- Fallback: mediana local

---

## Modelo

- **Algoritmo:** LightGBM (vencedor) / XGBoost
- **Target:** `log(price)` — previsão no espaço log, convertida com `expm1`
- **Validação:** 5-fold cross-validation out-of-fold
- **Métricas OOF (LightGBM):**
  - RMSE: R$643 (~19% da mediana de R$316)
  - MAE: R$59 (~19% da mediana)
- **Tracking:** MLflow local

---

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_eda.ipynb` | EDA de listings e calendário — distribuições, sazonalidade, comparação Jun/Set |
| `02_feature_engineering.ipynb` | Validação de features: HW, target encoding, reviews, geo, estimação de demanda e curvas de elasticidade |
| `02_reviews_analysis.ipynb` | Análise de sentimento, velocidade/recência, BERTopic |
| `03_modeling.ipynb` | XGBoost vs LightGBM, SHAP, análise de erros, pipeline de otimização, documentação de métricas |

---

## Deploy (Cloud Run)

```bash
# Build e push da imagem
docker build -f docker/api/Dockerfile -t airbnb-api:latest .
docker tag airbnb-api:latest us-central1-docker.pkg.dev/PROJECT_ID/airbnb-optimizer/api:latest
docker push us-central1-docker.pkg.dev/PROJECT_ID/airbnb-optimizer/api:latest

# Deploy
gcloud run deploy airbnb-price-api \
  --image=us-central1-docker.pkg.dev/PROJECT_ID/airbnb-optimizer/api:latest \
  --region=us-central1 \
  --allow-unauthenticated \
  --memory=2Gi --cpu=1 \
  --min-instances=0 --max-instances=3 \
  --port=8000
```

---

## Estrutura do Projeto

```
├── dags/                    # Airflow DAGs
├── data/                    # Dados (não versionados — DVC)
│   ├── raw/                 # CSV.gz do Inside Airbnb
│   └── processed/           # Parquet + artefatos de inferência
├── docker/                  # Dockerfiles
├── infra/                   # cloudrun.yaml, cloudbuild.yaml
├── models/                  # Modelo e encoders de produção
├── notebooks/               # EDA, feature engineering, modeling
├── scripts/                 # convert_to_parquet.py
├── src/
│   ├── features/            # Pipelines de features
│   ├── ingestion/           # Download Inside Airbnb
│   ├── serving/             # FastAPI (main, predict, schemas, security, settings)
│   └── training/            # Treino XGB/LGB + MLflow
├── requirements.txt         # Dependências completas (treino + features)
├── requirements-api.txt     # Dependências mínimas (serving)
└── docker-compose.yml       # Stack local completa
```

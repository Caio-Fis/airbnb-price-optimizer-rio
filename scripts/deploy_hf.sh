#!/usr/bin/env bash
# Deploy no Hugging Face Spaces (Docker Space).
#
# Monta .hf-space/ com o mínimo necessário (código + artefatos de runtime),
# cria o Space se não existir e faz push. Requer `hf auth login` prévio.
#
# Uso: ./scripts/deploy_hf.sh [usuario/nome-do-space]
set -euo pipefail

SPACE_ID="${1:-Caio-Fis/airbnb-price-optimizer-rio}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/.hf-space"

rm -rf "$BUILD"
mkdir -p "$BUILD/models" "$BUILD/data/processed" "$BUILD/data/external"

cp "$ROOT/Dockerfile" "$ROOT/requirements-api.txt" "$BUILD/"
rsync -a --exclude node_modules --exclude dist "$ROOT/frontend" "$BUILD/"
rsync -a --exclude __pycache__ "$ROOT/src" "$BUILD/"
cp "$ROOT/models/model.joblib" "$ROOT/models/encoders.joblib" "$BUILD/models/"
cp "$ROOT"/data/external/*.csv "$ROOT"/data/external/*.json "$BUILD/data/external/"
for f in feature_names competition_stats demand_params hw_seasonal_by_dow \
         prediction_intervals seasonal_factors; do
  cp "$ROOT/data/processed/$f.joblib" "$BUILD/data/processed/"
done
# clip_features.parquet fora: features de imagem desativadas (USE_IMAGE_FEATURES)
cp "$ROOT"/data/processed/geo_metro_cache.json "$ROOT"/data/processed/geo_poi_*.json \
   "$ROOT"/data/processed/listings_slim.parquet "$ROOT"/data/processed/listings_map.json.gz \
   "$BUILD/data/processed/"

# README com o front-matter exigido pelo Spaces
cat > "$BUILD/README.md" << 'EOF'
---
title: Airbnb Price Optimizer — Rio de Janeiro
emoji: 📍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Otimizador de Preço — Airbnb Rio de Janeiro

Clique em qualquer acomodação no mapa do Rio, escolha a data e descubra o preço
que maximiza a receita esperada (preço × ocupação), estimado por LightGBM +
curva de demanda por segmento + ajuste sazonal por dia (Carnaval, Réveillon,
fim de semana).

Código-fonte completo: https://github.com/Caio-Fis/airbnb-price-optimizer-rio
EOF

# Upload via API (funciona inclusive com tokens hf_oauth, que o git recusa)
python - << PYEOF
from huggingface_hub import HfApi, create_repo, upload_folder

# create_repo é paywalled para Docker Spaces em cpu-basic (402 Payment Required,
# exige PRO). Isso vale para CRIAR — um Space que já existe segue atualizável.
# Portanto: só tenta criar se ele realmente não existir, e deixa o 402 falar por
# si nesse caso, em vez de derrubar todo deploy de um Space existente.
api = HfApi()
try:
    api.space_info("${SPACE_ID}")
    print("Space já existe — pulando create_repo")
except Exception:
    create_repo("${SPACE_ID}", repo_type="space", space_sdk="docker", exist_ok=True)

upload_folder(
    folder_path="${BUILD}",
    repo_id="${SPACE_ID}",
    repo_type="space",
    commit_message="deploy $(date -u +%Y-%m-%dT%H:%M)",
    delete_patterns=["**"],
)
PYEOF
echo "✔ https://huggingface.co/spaces/${SPACE_ID}"

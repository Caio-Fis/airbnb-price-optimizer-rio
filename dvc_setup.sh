#!/bin/bash
# Script de setup do DVC com remote no GCS
# Execute: bash dvc_setup.sh

set -e

# Carregar variáveis do .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Inicializando DVC..."
pip install dvc[gs] --quiet
dvc init

echo "Configurando remote GCS..."
dvc remote add -d gcs_remote gs://${GCP_BUCKET_NAME}/dvc-cache
dvc remote modify gcs_remote credentialpath ${GOOGLE_APPLICATION_CREDENTIALS}

echo "Adicionando dados ao DVC..."
# Rastrear pastas de dados (não commitar os arquivos, só os .dvc pointers)
dvc add data/raw
dvc add data/images
dvc add data/processed
dvc add models

echo "Commitando configuração..."
git add .dvc/ .dvcignore dvc.yaml dvc_setup.sh data/.gitkeep 2>/dev/null || true
git add data/raw.dvc data/images.dvc data/processed.dvc models.dvc 2>/dev/null || true
git commit -m "feat: add DVC pipeline and GCS remote tracking"

echo ""
echo "DVC configurado! Para subir os dados:"
echo "  dvc push"
echo ""
echo "Para baixar os dados em outra máquina:"
echo "  dvc pull"

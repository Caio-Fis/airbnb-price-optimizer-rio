"""
Configuração do tracking do MLflow.

Existe por causa de um bug silencioso: o diretório deste projeto tem ESPAÇOS no
nome ("Airbnb value of homes"). Quando `MLFLOW_TRACKING_URI` não está definida,
o MLflow monta a URI default fazendo URL-quote do caminho local, produzindo

    sqlite:////home/crus/.../Airbnb%20value%20of%20homes/mlflow.db

O SQLAlchemy NÃO desfaz esse quoting: ele abre o caminho literal, com `%20` no
nome. O resultado é um banco-sombra em `Airbnb%20value%20of%20homes/mlflow.db`,
um diretório irmão criado sozinho. Os artefatos continuavam caindo em
`mlruns/<exp>/<run>/artifacts` (esse caminho vem do cwd, sem quoting), então
tudo parecia funcionar — mas métricas e parâmetros iam para o banco errado.
O `mlflow.db` de verdade ficou congelado em 12/Abr/2026 e 17 runs de Jul/2026
ficaram órfãos até serem recuperados por `scripts/migrate_mlflow_shadow_db.py`.

Regra: sempre configure o tracking por aqui, nunca confie no default.
"""
import os
from pathlib import Path

import mlflow
from loguru import logger

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "airbnb-price-optimizer")


def tracking_uri() -> str:
    """URI de tracking: `MLFLOW_TRACKING_URI` se definida, senão o SQLite local.

    A URI local é montada a partir do caminho absoluto SEM quoting — é isso que
    evita o banco-sombra descrito no topo do módulo.
    """
    explicit = os.getenv("MLFLOW_TRACKING_URI")
    if explicit:
        return explicit
    db_path = Path(os.getenv("MLFLOW_DB_PATH", "mlflow.db")).resolve()
    return f"sqlite:///{db_path}"


def configure_mlflow() -> str:
    """Aponta o MLflow para o store correto e devolve a URI usada."""
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)

    if "%" in uri:
        logger.warning(
            f"URI de tracking contém caracteres percent-encoded ({uri}) — "
            "provável banco-sombra. Ver src/training/mlflow_setup.py"
        )
    logger.info(f"MLflow tracking: {uri}")
    return uri

"""
Avaliação e promoção de modelo campeão no MLflow Model Registry.
"""
import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
from loguru import logger

from src.training.mlflow_setup import MLFLOW_EXPERIMENT_NAME, configure_mlflow

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
MODEL_REGISTRY_NAME = "airbnb-price-champion"
MODELS_PATH = Path("models")
BASELINE_METRICS_PATH = MODELS_PATH / "baseline_metrics.joblib"

# Um retreino sobre os mesmos dados oscila pouco; acima disso é regressão real.
# 1% sobre R$103 ≈ R$1, bem acima do ruído entre seeds observado (~R$0,6).
REGRESSION_TOLERANCE_PCT = float(os.getenv("CHAMPION_REGRESSION_TOLERANCE_PCT", "1.0"))


class ChampionRegressionError(RuntimeError):
    """O melhor modelo da rodada é pior que o campeão em produção."""


def _incumbent_rmse() -> float | None:
    """RMSE do campeão vigente, ou None se ainda não há campeão registrado."""
    if not BASELINE_METRICS_PATH.exists():
        return None
    try:
        return joblib.load(BASELINE_METRICS_PATH).get("oof_rmse")
    except Exception as e:  # arquivo corrompido não deve derrubar o pipeline
        logger.warning(f"baseline_metrics.joblib ilegível ({e}); guarda desativada")
        return None


def select_best_model(run_ids: list[str]) -> str:
    """Melhor run da rodada, desde que não seja pior que o campeão vigente.

    Sem essa comparação, `register_champion` promovia o melhor da rodada mesmo
    quando ele era pior que o modelo em produção — foi o que aconteceu em
    25/Jul/2026, quando um run de R$107,28 substituiu o campeão de R$103,73.
    Levanta ChampionRegressionError em vez de promover; defina
    CHAMPION_REGRESSION_TOLERANCE_PCT para afrouxar, ou compare manualmente.
    """
    configure_mlflow()
    client = mlflow.tracking.MlflowClient()
    best_run_id = None
    best_rmse = float("inf")

    for run_id in run_ids:
        if run_id is None:
            continue
        run = client.get_run(run_id)
        rmse = run.data.metrics.get("oof_rmse", float("inf"))
        logger.info(f"Run {run_id}: OOF RMSE = R${rmse:.2f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_run_id = run_id

    if best_run_id is None:
        raise ChampionRegressionError("Nenhum run válido para avaliar")

    logger.info(f"Best model: run_id={best_run_id} | RMSE=R${best_rmse:.2f}")

    incumbent = _incumbent_rmse()
    if incumbent is None:
        logger.info("Sem campeão vigente — promovendo sem comparação")
        return best_run_id

    limit = incumbent * (1 + REGRESSION_TOLERANCE_PCT / 100)
    if best_rmse > limit:
        raise ChampionRegressionError(
            f"Regressão bloqueada: melhor da rodada R${best_rmse:.2f} > "
            f"limite R${limit:.2f} (campeão R${incumbent:.2f} "
            f"+{REGRESSION_TOLERANCE_PCT}%). Campeão em produção mantido."
        )

    delta = best_rmse - incumbent
    logger.info(
        f"Guarda de regressão OK: R${best_rmse:.2f} vs campeão R${incumbent:.2f} "
        f"({delta:+.2f})"
    )
    return best_run_id


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def _install_artifact(src: Path, dst: Path, label: str) -> None:
    """Copia src→dst deixando rastro: hashes antes/depois e backup do anterior.

    O encoder do serving já divergiu do encoder do treino sem ninguém notar
    (26/Jul/2026: `neighbourhood_enc` fora por até 0,07 em log-space e
    `price_p99` ausente, o que matou a proteção contra input outlier). A troca
    era um `shutil.copy` mudo — agora ela aparece no log e o anterior fica
    recuperável.
    """
    if not src.exists():
        logger.warning(f"{label}: origem {src} não existe, mantendo o atual")
        return

    new_hash = _md5(src)
    if dst.exists():
        old_hash = _md5(dst)
        if old_hash == new_hash:
            logger.info(f"{label}: inalterado (md5 {new_hash})")
            return
        backup = dst.with_suffix(f"{dst.suffix}.bak-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(dst, backup)
        logger.warning(
            f"{label}: SUBSTITUÍDO — md5 {old_hash} → {new_hash}. "
            f"Anterior salvo em {backup.name}"
        )
    else:
        logger.info(f"{label}: instalado pela primeira vez (md5 {new_hash})")
    shutil.copy2(src, dst)


def register_champion(run_id: str) -> None:
    configure_mlflow()
    client = mlflow.tracking.MlflowClient()

    # Registrar no Model Registry
    model_uri = f"runs:/{run_id}/sklearn_model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
    logger.info(f"Model registered: {MODEL_REGISTRY_NAME} v{mv.version}")

    # Promover para Production (depreca versões antigas)
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(f"Model v{mv.version} promoted to Production")

    # Baixar modelo e salvar localmente para a API
    model_path = MODELS_PATH
    model_path.mkdir(exist_ok=True)
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(loaded_model, model_path / "model.joblib")
    except Exception as e:
        logger.warning(f"MLflow model download skipped ({e}); copying from processed data")
        for fname in ["model_xgboost.joblib", "model_lightgbm.joblib"]:
            src = PROCESSED_DATA_PATH / fname
            if src.exists():
                _install_artifact(src, model_path / "model.joblib", "model.joblib")
                break

    # Encoders: o serving precisa exatamente do encoder desta rodada de treino
    _install_artifact(
        PROCESSED_DATA_PATH / "encoders.joblib",
        model_path / "encoders.joblib",
        "encoders.joblib",
    )

    # Salvar métricas do campeão para o monitoramento de drift
    run = client.get_run(run_id)
    champion_rmse = run.data.metrics.get("oof_rmse")
    if champion_rmse is not None:
        joblib.dump(
            {"oof_rmse": champion_rmse, "registered_at": str(datetime.now(UTC))},
            BASELINE_METRICS_PATH,
        )
        logger.info(f"Baseline metrics saved: RMSE=R${champion_rmse:.2f}")

    logger.info("Model and encoders saved to models/ for API serving")

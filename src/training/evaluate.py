"""
Avaliação e promoção de modelo campeão no MLflow Model Registry.
"""
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "airbnb-price-optimizer")
MODEL_REGISTRY_NAME = "airbnb-price-champion"


def select_best_model(run_ids: list[str]) -> str:
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

    logger.info(f"Best model: run_id={best_run_id} | RMSE=R${best_rmse:.2f}")
    return best_run_id


def register_champion(run_id: str) -> None:
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
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(loaded_model, model_path / "model.joblib")
    except Exception as e:
        logger.warning(f"MLflow model download skipped ({e}); copying from processed data")
        import shutil
        for fname in [f"model_xgboost.joblib", f"model_lightgbm.joblib"]:
            src = PROCESSED_DATA_PATH / fname
            if src.exists():
                shutil.copy(src, model_path / "model.joblib")
                break

    # Copiar encoders
    encoders_src = PROCESSED_DATA_PATH / "encoders.joblib"
    if encoders_src.exists():
        import shutil
        shutil.copy(encoders_src, model_path / "encoders.joblib")

    # Salvar métricas do campeão para o monitoramento de drift
    run = client.get_run(run_id)
    champion_rmse = run.data.metrics.get("oof_rmse")
    if champion_rmse is not None:
        joblib.dump(
            {"oof_rmse": champion_rmse, "registered_at": str(datetime.utcnow())},
            model_path / "baseline_metrics.joblib",
        )
        logger.info(f"Baseline metrics saved: RMSE=R${champion_rmse:.2f}")

    logger.info("Model and encoders saved to models/ for API serving")

"""
Monitoramento com Evidently AI:
- Data drift: detecta mudanças na distribuição das features
- Performance: compara RMSE atual vs baseline do treinamento
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
RMSE_THRESHOLD = float(os.getenv("RMSE_THRESHOLD", "1.2"))  # 20% de degradação permitida


def _load_baseline_rmse() -> float:
    """Carrega RMSE do campeão registrado; fallback em env var."""
    path = Path("models/baseline_metrics.joblib")
    if path.exists():
        try:
            return float(joblib.load(path)["oof_rmse"])
        except Exception:
            pass
    return float(os.getenv("BASELINE_RMSE", "50.0"))


BASELINE_RMSE = _load_baseline_rmse()


class DriftDetector:
    """Detecta data drift comparando distribuição atual vs referência."""

    def run(self) -> dict:
        reference = self._load_reference()
        current = self._load_current()

        if reference is None or current is None:
            logger.warning("Cannot compute drift: missing reference or current data")
            return {"dataset_drift": False, "drift_share": 0.0}

        numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["log_price", "price"]]

        column_mapping = ColumnMapping(numerical_features=numeric_cols)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

        result = report.as_dict()
        drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
        dataset_drift = drift_share > 0.3  # drift se >30% das colunas mudaram

        logger.info(f"Drift share: {drift_share:.2%} | Drift detected: {dataset_drift}")

        # Salvar relatório
        report_path = PROCESSED_DATA_PATH / f"drift_report_{datetime.now().date()}.html"
        report.save_html(str(report_path))

        return {"dataset_drift": dataset_drift, "drift_share": drift_share}

    def _load_reference(self) -> pd.DataFrame | None:
        path = PROCESSED_DATA_PATH / "final_features.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        # Usar 80% antigo como referência
        return df.iloc[: int(len(df) * 0.8)]

    def _load_current(self) -> pd.DataFrame | None:
        path = PROCESSED_DATA_PATH / "final_features.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        # Usar 20% mais recente como current
        return df.iloc[int(len(df) * 0.8) :]


class PerformanceMonitor:
    """Verifica degradação de RMSE no modelo em produção."""

    def run(self) -> dict:
        model_path = Path("models/model.joblib")
        if not model_path.exists():
            logger.warning("No model found for performance monitoring")
            return {"rmse_degraded": False}

        model = joblib.load(model_path)
        features_path = PROCESSED_DATA_PATH / "final_features.parquet"
        if not features_path.exists():
            return {"rmse_degraded": False}

        df = pd.read_parquet(features_path)
        # Avaliar nos dados mais recentes (últimos 20%)
        df = df.iloc[int(len(df) * 0.8) :]

        feature_names_path = PROCESSED_DATA_PATH / "feature_names.joblib"
        if feature_names_path.exists():
            feature_names = joblib.load(feature_names_path)
            X = df[feature_names].select_dtypes(include=[np.number])
        else:
            X = df.select_dtypes(include=[np.number]).drop(
                columns=["log_price", "price"], errors="ignore"
            )

        y_true = df["log_price"].values
        y_pred = model.predict(X)

        rmse = float(np.sqrt(np.mean((np.expm1(y_true) - np.expm1(y_pred)) ** 2)))
        degraded = rmse > BASELINE_RMSE * RMSE_THRESHOLD

        logger.info(
            f"Current RMSE: R${rmse:.2f} | Baseline: R${BASELINE_RMSE:.2f} | "
            f"Threshold: R${BASELINE_RMSE * RMSE_THRESHOLD:.2f} | Degraded: {degraded}"
        )

        return {"rmse_degraded": degraded, "current_rmse": rmse, "baseline_rmse": BASELINE_RMSE}

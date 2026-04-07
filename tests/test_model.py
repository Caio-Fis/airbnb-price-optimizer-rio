"""Testes para treinamento e avaliação do modelo."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestTraining:
    def _make_features(self, n=200):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(n, 20), columns=[f"feature_{i}" for i in range(20)])
        y = np.log1p(200 + 50 * X["feature_0"] + np.random.normal(0, 10, n))
        y = pd.Series(y, name="log_price")
        df = pd.concat([X, y], axis=1)
        df["log_price"] = y
        return df

    def test_rmse_original_space(self):
        from src.training.train import _rmse_original_space
        y_true = np.array([np.log1p(200), np.log1p(300)])
        y_pred = np.array([np.log1p(210), np.log1p(290)])
        rmse = _rmse_original_space(y_true, y_pred)
        assert rmse > 0
        assert rmse < 50  # erro pequeno para previsões próximas

    def test_mae_original_space(self):
        from src.training.train import _mae_original_space
        y_true = np.array([np.log1p(200), np.log1p(300)])
        y_pred = np.array([np.log1p(200), np.log1p(300)])
        mae = _mae_original_space(y_true, y_pred)
        assert mae == pytest.approx(0.0, abs=1e-5)

    def test_exclude_cols_are_dropped(self):
        from src.training import train as train_module
        exclude = train_module.EXCLUDE_COLS
        assert "price" in exclude
        assert "log_price" in exclude
        assert "id" in exclude


class TestEvaluate:
    def test_select_best_model_picks_lower_rmse(self):
        from src.training.evaluate import select_best_model

        mock_client = MagicMock()
        run_a = MagicMock()
        run_a.data.metrics = {"oof_rmse": 80.0}
        run_b = MagicMock()
        run_b.data.metrics = {"oof_rmse": 55.0}

        mock_client.get_run.side_effect = lambda run_id: {
            "run_a": run_a, "run_b": run_b
        }[run_id]

        with patch("src.training.evaluate.mlflow.tracking.MlflowClient", return_value=mock_client):
            best = select_best_model(["run_a", "run_b"])

        assert best == "run_b"

    def test_select_best_model_handles_none(self):
        from src.training.evaluate import select_best_model

        mock_client = MagicMock()
        run_a = MagicMock()
        run_a.data.metrics = {"oof_rmse": 60.0}
        mock_client.get_run.return_value = run_a

        with patch("src.training.evaluate.mlflow.tracking.MlflowClient", return_value=mock_client):
            best = select_best_model([None, "run_a"])

        assert best == "run_a"

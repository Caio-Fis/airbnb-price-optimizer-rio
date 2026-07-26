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


class TestRegressionGuard:
    """A promoção do campeão comparava apenas os runs da própria rodada, então
    um retreino ruim virava Production sozinho — aconteceu em 25/Jul/2026, quando
    um modelo de R$107,28 substituiu o campeão de R$103,73. Estes testes travam
    a guarda que passou a comparar contra o campeão vigente."""

    def _select(self, rmse: float, incumbent: float | None):
        from src.training.evaluate import select_best_model

        run = MagicMock()
        run.data.metrics = {"oof_rmse": rmse}
        mock_client = MagicMock()
        mock_client.get_run.return_value = run

        with patch("src.training.evaluate.mlflow.tracking.MlflowClient", return_value=mock_client), \
             patch("src.training.evaluate._incumbent_rmse", return_value=incumbent):
            return select_best_model(["run_a"])

    def test_bloqueia_modelo_pior_que_o_campeao(self):
        from src.training.evaluate import ChampionRegressionError

        # O caso real: 107,28 contra o campeão de 103,73.
        with pytest.raises(ChampionRegressionError, match="Regressão bloqueada"):
            self._select(rmse=107.28, incumbent=103.73)

    def test_promove_modelo_melhor(self):
        assert self._select(rmse=102.95, incumbent=103.73) == "run_a"

    def test_tolera_piora_dentro_da_margem(self):
        # 103,90 está a +0,16% do campeão — dentro da tolerância de 1%.
        assert self._select(rmse=103.90, incumbent=103.73) == "run_a"

    def test_promove_quando_nao_ha_campeao(self):
        assert self._select(rmse=500.0, incumbent=None) == "run_a"

    def test_incumbent_rmse_le_o_arquivo_do_campeao(self, tmp_path):
        import joblib as jl
        from src.training import evaluate

        path = tmp_path / "baseline_metrics.joblib"
        jl.dump({"oof_rmse": 99.5}, path)
        with patch.object(evaluate, "BASELINE_METRICS_PATH", path):
            assert evaluate._incumbent_rmse() == 99.5

    def test_incumbent_ausente_nao_derruba_o_pipeline(self, tmp_path):
        from src.training import evaluate

        with patch.object(evaluate, "BASELINE_METRICS_PATH", tmp_path / "nao_existe.joblib"):
            assert evaluate._incumbent_rmse() is None


class TestInstallArtifact:
    """`register_champion` sobrescrevia models/encoders.joblib com um shutil.copy
    mudo — foi assim que o serving rodou com um encoder diferente do treino sem
    alarme. Agora a troca loga os hashes e faz backup do anterior."""

    def test_faz_backup_quando_o_conteudo_muda(self, tmp_path):
        from src.training.evaluate import _install_artifact

        src, dst = tmp_path / "src.joblib", tmp_path / "dst.joblib"
        src.write_bytes(b"encoder-novo")
        dst.write_bytes(b"encoder-antigo")

        _install_artifact(src, dst, "encoders.joblib")

        assert dst.read_bytes() == b"encoder-novo"
        backups = list(tmp_path.glob("dst.joblib.bak-*"))
        assert len(backups) == 1, "o encoder anterior precisa ficar recuperável"
        assert backups[0].read_bytes() == b"encoder-antigo"

    def test_nao_faz_backup_quando_identico(self, tmp_path):
        from src.training.evaluate import _install_artifact

        src, dst = tmp_path / "src.joblib", tmp_path / "dst.joblib"
        src.write_bytes(b"igual")
        dst.write_bytes(b"igual")

        _install_artifact(src, dst, "encoders.joblib")

        assert list(tmp_path.glob("dst.joblib.bak-*")) == []

    def test_origem_ausente_preserva_o_destino(self, tmp_path):
        from src.training.evaluate import _install_artifact

        src, dst = tmp_path / "nao_existe.joblib", tmp_path / "dst.joblib"
        dst.write_bytes(b"campeao-atual")

        _install_artifact(src, dst, "encoders.joblib")

        assert dst.read_bytes() == b"campeao-atual"


class TestMlflowSetup:
    """O nome do diretório do projeto tem espaços. O default do MLflow fazia
    URL-quote do caminho e o SQLAlchemy abria o caminho literal, criando um
    banco-sombra em `Airbnb%20value%20of%20homes/mlflow.db` — 17 runs foram
    parar lá. A URI precisa sair sem percent-encoding."""

    def test_uri_local_nao_tem_percent_encoding(self, tmp_path, monkeypatch):
        from src.training import mlflow_setup

        db = tmp_path / "dir com espacos" / "mlflow.db"
        db.parent.mkdir()
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MLFLOW_DB_PATH", str(db))

        uri = mlflow_setup.tracking_uri()

        assert "%20" not in uri, "percent-encoding recria o banco-sombra"
        assert uri == f"sqlite:///{db}"
        assert "dir com espacos" in uri

    def test_env_explicita_tem_precedencia(self, monkeypatch):
        from src.training import mlflow_setup

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        assert mlflow_setup.tracking_uri() == "http://localhost:5000"

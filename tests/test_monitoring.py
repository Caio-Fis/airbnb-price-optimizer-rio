"""Testes do monitoramento.

O Evidently é importado sob demanda dentro de `DriftDetector.run()`, então tudo
que está aqui roda sem a dependência: a seleção de colunas (onde estava o bug),
o parsing de preço e os caminhos de artefato ausente.
"""
import numpy as np
import pandas as pd
import pytest

from src.monitoring import drift as _drift


def _drift_module():
    return _drift


@pytest.fixture
def snapshots():
    """Dois snapshots sintéticos com drift conhecido em uma coluna."""
    rng = np.random.default_rng(0)
    n = 500
    ref = pd.DataFrame({
        "id": np.arange(n),
        "scrape_id": np.full(n, 1),
        "host_id": rng.integers(0, 1000, n),
        "accommodates": rng.integers(1, 8, n),
        "availability_365": rng.integers(0, 200, n),   # driftada de propósito
        "license": [None] * n,                          # toda nula
        "bairro": ["Copacabana"] * n,                   # não-numérica
        "price": rng.normal(300, 50, n),
    })
    cur = ref.copy()
    cur["availability_365"] = rng.integers(200, 365, n)
    return ref, cur


class TestSelecaoDeColunas:
    """O bug estava aqui: quais colunas entram na comparação."""

    def test_descarta_identificadores_nulas_e_nao_numericas(self, snapshots):
        drift = _drift_module()
        ref, cur = snapshots
        cols, target = drift.DriftDetector._select_columns(ref, cur)

        for proibida in ("id", "scrape_id", "host_id", "license", "bairro"):
            assert proibida not in cols, f"{proibida} não deveria entrar"
        assert "accommodates" in cols
        assert "availability_365" in cols

    def test_price_vira_target_e_nao_feature(self, snapshots):
        drift = _drift_module()
        ref, cur = snapshots
        cols, target = drift.DriftDetector._select_columns(ref, cur)
        assert target == "price"
        assert "price" not in cols, "price é target, não feature"

    def test_sem_price_valido_nao_ha_target(self, snapshots):
        drift = _drift_module()
        ref, cur = snapshots
        ref = ref.assign(price=np.nan)
        cols, target = drift.DriftDetector._select_columns(ref, cur)
        assert target is None
        assert cols, "as demais features continuam valendo"

    def test_so_entra_coluna_presente_nos_dois(self, snapshots):
        drift = _drift_module()
        ref, cur = snapshots
        cols, _ = drift.DriftDetector._select_columns(ref, cur.drop(columns=["accommodates"]))
        assert "accommodates" not in cols


class TestRegressaoDoBugAntigo:
    """Garante que o detector não volte a comparar um snapshot consigo mesmo."""

    def test_referencia_e_atual_sao_arquivos_diferentes(self):
        drift = _drift_module()
        assert drift.REFERENCE_SNAPSHOT != drift.CURRENT_SNAPSHOT, (
            "referência e atual apontando para o mesmo arquivo — é o bug antigo, "
            "que comparava os primeiros 80% contra os últimos 20% do mesmo scrape "
            "e acusava drift em toda execução"
        )
        assert "jun" in drift.REFERENCE_SNAPSHOT.lower()
        assert "set" in drift.CURRENT_SNAPSHOT.lower()

    def test_snapshot_ausente_nao_derruba(self, monkeypatch, tmp_path):
        drift = _drift_module()
        monkeypatch.setattr(drift, "PROCESSED_DATA_PATH", tmp_path)
        resultado = drift.DriftDetector().run()
        assert resultado["dataset_drift"] is False
        assert resultado["drift_share"] == 0.0


class TestPrecoParseia:
    def test_price_string_com_cifrao_vira_numero(self, monkeypatch, tmp_path):
        drift = _drift_module()
        df = pd.DataFrame({"id": [1, 2, 3], "price": ["$1,234.00", "None", "$99.50"]})
        df.to_parquet(tmp_path / "s.parquet")
        monkeypatch.setattr(drift, "PROCESSED_DATA_PATH", tmp_path)
        out = drift._read_snapshot("s.parquet")
        assert out["price"].tolist()[0] == 1234.0
        assert out["price"].tolist()[2] == 99.5
        assert pd.isna(out["price"].tolist()[1]), "'None' vira NaN, não crash"


class TestSanidadeDoModelo:
    def test_alias_antigo_preservado(self):
        """O monitoring_dag importa PerformanceMonitor pelo nome antigo."""
        drift = _drift_module()
        assert drift.PerformanceMonitor is drift.ModelSanityCheck

    def test_sem_modelo_nao_derruba(self, monkeypatch, tmp_path):
        drift = _drift_module()
        monkeypatch.setattr(drift, "PROCESSED_DATA_PATH", tmp_path)
        monkeypatch.chdir(tmp_path)
        resultado = drift.ModelSanityCheck().run()
        assert resultado["model_broken"] is False
        assert resultado["checked"] is False


class TestContratoComODag:
    """O monitoring_dag lê chaves específicas do dicionário de retorno.

    Foi exatamente aqui que a renomeação quebrou: o DAG lia `rmse_degraded` e o
    módulo passou a devolver `model_broken` — KeyError só no Airflow, nunca no
    pytest. Este teste prende o contrato.
    """

    CHAVES_DRIFT = {"dataset_drift", "drift_share"}
    CHAVES_SANIDADE = {"model_broken"}

    def test_drift_devolve_as_chaves_que_o_dag_le(self, monkeypatch, tmp_path):
        drift = _drift_module()
        monkeypatch.setattr(drift, "PROCESSED_DATA_PATH", tmp_path)
        assert self.CHAVES_DRIFT <= set(drift.DriftDetector().run())

    def test_sanidade_devolve_as_chaves_que_o_dag_le(self, monkeypatch, tmp_path):
        drift = _drift_module()
        monkeypatch.setattr(drift, "PROCESSED_DATA_PATH", tmp_path)
        monkeypatch.chdir(tmp_path)
        assert self.CHAVES_SANIDADE <= set(drift.ModelSanityCheck().run())

    def test_dag_usa_branch_para_decidir_retreino(self):
        """PythonOperator ignora o task_id devolvido: os dois ramos rodavam."""
        fonte = (
            __import__("pathlib").Path("dags/monitoring_dag.py").read_text()
        )
        assert "BranchPythonOperator(" in fonte
        i = fonte.index('task_id="decide_retraining"')
        assert "BranchPythonOperator" in fonte[max(0, i - 300):i]

    def test_chaves_lidas_pelo_dag_batem_com_o_modulo(self):
        """Varre o DAG atrás de xcom_push/report[...] e confere contra o módulo."""
        import re
        from pathlib import Path

        fonte = Path("dags/monitoring_dag.py").read_text()
        lidas = set(re.findall(r'report\["([^"]+)"\]', fonte))
        conhecidas = self.CHAVES_DRIFT | self.CHAVES_SANIDADE
        assert lidas <= conhecidas, f"o DAG lê chaves que o módulo não devolve: {lidas - conhecidas}"

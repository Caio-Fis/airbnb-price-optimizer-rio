"""
Monitoramento com Evidently AI:
- Data drift: compara os snapshots Jun/2025 e Set/2025 do Inside Airbnb
- Sanidade do modelo: confere que o campeão ainda produz previsões coerentes

NOTA SOBRE O QUE ESTE MÓDULO PODE E NÃO PODE DETECTAR
-----------------------------------------------------
`DriftDetector` mede drift real: são dois scrapes de datas diferentes.

`ModelSanityCheck` NÃO mede degradação de performance. Não existe conjunto
rotulado fora da amostra: o campeão foi treinado com todo o `final_features`,
então qualquer RMSE calculado aqui é IN-SAMPLE e sempre sairá muito abaixo do
OOF. Monitorar degradação de verdade exigiria preços observados posteriores ao
treino, que este projeto não coleta. O que dá para detectar é quebra grosseira
(artefato corrompido, schema de features divergente, modelo trocado) — e é isso
que a classe faz, com o nome dizendo a verdade.
"""
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

# O Evidently só está no ambiente de treino (requirements.txt), não no da API
# (requirements-api.txt). Importar sob demanda deixa o módulo carregável nos dois
# — só `DriftDetector.run()` precisa dele. A seleção de colunas, que é onde
# estava o bug, fica testável sem a dependência pesada.

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
REPORT_PATH = Path(os.getenv("DRIFT_REPORT_PATH", "docs/evidence"))

# Os dois scrapes do Inside Airbnb para o RJ. Jun é a referência (passado),
# Set é o atual — é esta comparação que caracteriza drift temporal.
REFERENCE_SNAPSHOT = os.getenv("DRIFT_REFERENCE", "listings_jun2025.parquet")
CURRENT_SNAPSHOT = os.getenv("DRIFT_CURRENT", "listings_set2025.parquet")

DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.3"))
SANITY_RMSE_TOLERANCE = float(os.getenv("SANITY_RMSE_TOLERANCE", "1.2"))

# Identificadores e metadados de scrape: variam entre snapshots por construção e
# acusariam drift sem significado nenhum.
NON_FEATURE_COLS = {
    "id",
    "scrape_id",
    "host_id",
    "calendar_updated",
    "license",
    "neighbourhood_group_cleansed",
}


def _load_baseline_rmse() -> float:
    """RMSE out-of-fold do campeão registrado; fallback em env var."""
    path = Path("models/baseline_metrics.joblib")
    if path.exists():
        try:
            return float(joblib.load(path)["oof_rmse"])
        except Exception:
            logger.warning("baseline_metrics.joblib ilegível; usando BASELINE_RMSE")
    return float(os.getenv("BASELINE_RMSE", "50.0"))


BASELINE_RMSE = _load_baseline_rmse()


def _read_snapshot(filename: str) -> pd.DataFrame | None:
    path = PROCESSED_DATA_PATH / filename
    if not path.exists():
        logger.warning(f"Snapshot ausente: {path}")
        return None
    df = pd.read_parquet(path)
    # `price` vem como string ('$1,234.00' ou 'None') neste scrape.
    if "price" in df.columns:
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
    return df


class DriftDetector:
    """Drift entre os snapshots de Jun/2025 e Set/2025.

    A versão anterior comparava os primeiros 80% das linhas contra os últimos 20%
    do MESMO arquivo — um snapshot só, sem eixo temporal. E como `final_features`
    sai grosso modo ordenado por `id`, que cresce com a data de cadastro, isso
    comparava anúncios ANTIGOS (mediana 13 reviews) contra RECÉM-CADASTRADOS
    (mediana 1 review).

    Na prática ela nem chegava a rodar: `select_dtypes(number)` incluía
    `neighbourhood_group_cleansed`, `calendar_updated` e `license`, que são
    numéricas e 100% nulas, e o Evidently levanta ValueError em coluna vazia. A
    task de drift do Airflow falhava em toda execução. Removendo as vazias para
    poder medir, dava 44,3% (47/106) — acima do limite de 30%, ou seja, alarme
    positivo em todo run, independentemente do mercado.

    Medido com os snapshots de verdade (Evidently 0.4.30, Wasserstein normed):
    **3/40 colunas driftadas = 7,5%, drift=False**. As três são mecânicas em 3
    meses de distância: availability_60 (0,107), availability_90 (0,113) e
    availability_eoy (0,918 — os dias restantes até o fim do ano encolhem por
    definição). O `price`, como target, NÃO driftou: score 0,0089.
    """

    def run(self) -> dict:
        reference = self._load_reference()
        current = self._load_current()

        if reference is None or current is None:
            logger.warning("Sem drift: falta o snapshot de referência ou o atual")
            return {"dataset_drift": False, "drift_share": 0.0}

        feature_cols, target_col = self._select_columns(reference, current)
        if not feature_cols:
            logger.warning("Sem colunas numéricas comuns aos dois snapshots")
            return {"dataset_drift": False, "drift_share": 0.0}

        # Só agora: sem dado, não há por que pagar o import pesado.
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report

        cols = feature_cols + ([target_col] if target_col else [])
        reference = reference[cols]
        current = current[cols]

        column_mapping = ColumnMapping(
            numerical_features=feature_cols,
            target=target_col,
        )
        metrics = [DataDriftPreset()]
        if target_col:
            metrics.append(TargetDriftPreset())

        report = Report(metrics=metrics)
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping,
        )

        result = report.as_dict()
        drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
        dataset_drift = drift_share > DRIFT_SHARE_THRESHOLD

        logger.info(
            f"Drift Jun→Set | referência={len(reference)} atual={len(current)} "
            f"| colunas={len(feature_cols)} | share={drift_share:.2%} "
            f"| drift={dataset_drift}"
        )

        REPORT_PATH.mkdir(parents=True, exist_ok=True)
        # Nome estável: o relatório é versionado no repo e substitui o link do
        # GCS que morreu junto com o projeto.
        report.save_html(str(REPORT_PATH / "drift_report_jun_set_2025.html"))
        report.save_html(
            str(REPORT_PATH / f"drift_report_{datetime.now().date()}.html")
        )

        return {"dataset_drift": dataset_drift, "drift_share": drift_share}

    @staticmethod
    def _select_columns(
        reference: pd.DataFrame, current: pd.DataFrame
    ) -> tuple[list[str], str | None]:
        """Numéricas presentes nos dois snapshots, sem identificadores nem colunas
        totalmente nulas (que o Evidently não consegue testar)."""
        comuns = [c for c in reference.columns if c in current.columns]
        cols = []
        for c in comuns:
            if c in NON_FEATURE_COLS or c == "price":
                continue
            if not (
                pd.api.types.is_numeric_dtype(reference[c])
                and pd.api.types.is_numeric_dtype(current[c])
            ):
                continue
            if reference[c].isna().all() or current[c].isna().all():
                continue
            cols.append(c)

        # `price` entra como TARGET, não como feature: o deslocamento do preço de
        # mercado é o sinal principal, e separá-lo evita diluí-lo entre as demais.
        target = None
        if "price" in comuns and not reference["price"].isna().all():
            target = "price"
        return cols, target

    def _load_reference(self) -> pd.DataFrame | None:
        return _read_snapshot(REFERENCE_SNAPSHOT)

    def _load_current(self) -> pd.DataFrame | None:
        return _read_snapshot(CURRENT_SNAPSHOT)


class ModelSanityCheck:
    """Confere que o campeão carrega e prevê de forma coerente.

    NÃO é monitoramento de degradação — ver a nota no topo do módulo. O RMSE
    calculado aqui é in-sample e por isso fica bem ABAIXO do OOF; a checagem
    dispara só quando ele passa do baseline OOF, o que em dados de treino
    significa quebra grosseira (modelo trocado, features desalinhadas,
    artefato corrompido), não desgaste de mercado.
    """

    def run(self) -> dict:
        model_path = Path("models/model.joblib")
        if not model_path.exists():
            logger.warning("Nenhum modelo em models/model.joblib")
            return {"model_broken": False, "checked": False}

        features_path = PROCESSED_DATA_PATH / "final_features.parquet"
        if not features_path.exists():
            logger.warning("final_features.parquet ausente")
            return {"model_broken": False, "checked": False}

        model = joblib.load(model_path)
        df = pd.read_parquet(features_path)

        feature_names_path = PROCESSED_DATA_PATH / "feature_names.joblib"
        if feature_names_path.exists():
            feature_names = joblib.load(feature_names_path)
            faltando = [c for c in feature_names if c not in df.columns]
            if faltando:
                logger.error(f"Schema divergente: {len(faltando)} features ausentes")
                return {"model_broken": True, "checked": True, "missing": faltando}
            X = df[feature_names]
        else:
            X = df.select_dtypes(include=[np.number]).drop(
                columns=["log_price", "price"], errors="ignore"
            )

        y_true = df["log_price"].values
        y_pred = model.predict(X)

        rmse = float(np.sqrt(np.mean((np.expm1(y_true) - np.expm1(y_pred)) ** 2)))
        broken = rmse > BASELINE_RMSE * SANITY_RMSE_TOLERANCE

        logger.info(
            f"Sanidade do modelo | RMSE in-sample: R${rmse:.2f} "
            f"| baseline OOF: R${BASELINE_RMSE:.2f} "
            f"| limite: R${BASELINE_RMSE * SANITY_RMSE_TOLERANCE:.2f} "
            f"| quebrado: {broken}"
        )
        if not broken and rmse > BASELINE_RMSE:
            logger.warning(
                "RMSE in-sample acima do OOF — inesperado, vale investigar "
                "antes que vire quebra"
            )

        return {
            "model_broken": broken,
            "checked": True,
            "in_sample_rmse": rmse,
            "baseline_oof_rmse": BASELINE_RMSE,
        }


# Nome antigo mantido para não quebrar o monitoring_dag; o comportamento agora
# é o da checagem de sanidade, que é o que a classe sempre fez de fato.
PerformanceMonitor = ModelSanityCheck

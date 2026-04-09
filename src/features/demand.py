"""
Módulo de otimização de receita via curva de demanda.

Abordagem: elasticidade de preço estimada por painel de snapshots.
  - Para cada listing com dados em dois snapshots: Δlogit(occ) ~ b * Δlog(price)
  - Controla efeitos fixos por listing (qualidade, localização) e sazonalidade
  - Segmentos sem variação suficiente usam fallback cross-sectional
  - Com múltiplos pares de snapshots: weighted average de b (pares recentes > peso)

Preço ótimo: argmax[ price × sigmoid(a + b × log(price / comp_p50)) ]

Saídas:
  demand_features.parquet  — por listing: occupancy, elasticidade, preço ótimo
  demand_params.joblib     — por segmento: parâmetros da curva de demanda
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import expit, logit as scipy_logit
from sklearn.linear_model import LinearRegression

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))

JUN_SCRAPE = pd.Timestamp("2025-06-24")
SET_SCRAPE = pd.Timestamp("2025-09-26")
WINDOW_DAYS = 90

MIN_OCCUPANCY = 0.03       # listing ativo: mínimo 3% no período
MIN_SEGMENT_PANEL = 15     # mín. de listings com variação de preço para painel
MIN_SEGMENT_CROSS = 10     # mín. para fallback cross-sectional

# Grid para busca do preço ótimo (multiplicadores do comp_p50)
PRICE_GRID = np.linspace(0.3, 4.0, 500)
MAX_OPTIMAL_MULT = 3.5     # cap de segurança


def _occupancy(path: Path, scrape_date: pd.Timestamp) -> pd.Series:
    """Fração de dias NÃO disponíveis nos WINDOW_DAYS dias após o scrape."""
    end = scrape_date + pd.Timedelta(days=WINDOW_DAYS)
    df = pd.read_parquet(path, columns=["listing_id", "date", "available"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= scrape_date) & (df["date"] < end)]
    return (
        df.groupby("listing_id")["available"]
        .apply(lambda x: 1.0 - (x == "t").mean())
        .rename("occupancy")
    )


def _safe_logit(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    return scipy_logit(np.clip(x, eps, 1 - eps))


def _optimal_price(
    a: float, b: float, comp_p50: float, comp_p75: float
) -> tuple[float, float, str]:
    """
    Retorna (optimal_price, expected_occupancy, strategy).

    b < -1: máximo interior real — revenue-maximizing price.
    -1 ≤ b < 0: demanda inelástica — sem máximo interior; recomenda P75 do segmento
                (subir até P75 aumenta receita sem penalizar ocupação significativamente).
    """
    if b < -1:
        # Interior optimum: P* = (1 + 1/b) para logística normalizada
        # Confirma via grid search
        revenues = PRICE_GRID * comp_p50 * expit(a + b * np.log(np.maximum(PRICE_GRID, 1e-6)))
        best = int(np.argmax(revenues))
        opt_mult = min(PRICE_GRID[best], MAX_OPTIMAL_MULT)
        opt_price = opt_mult * comp_p50
        opt_occ = float(expit(a + b * np.log(opt_mult)))
        strategy = "revenue_optimal"
    else:
        # Sem máximo interior — demanda inelástica: posicionar no P75
        opt_price = float(comp_p75) if pd.notna(comp_p75) and comp_p75 > comp_p50 else comp_p50 * 1.25
        opt_mult = opt_price / comp_p50 if comp_p50 > 0 else 1.0
        opt_occ = float(expit(a + b * np.log(max(opt_mult, 0.1))))
        strategy = "premium_positioning"  # subir até P75 é seguro

    return opt_price, opt_occ, strategy


class DemandPipeline:
    """
    Pipeline de estimação de curva de demanda.

    snapshot_pairs: lista de dicts com chaves:
        - listings_earlier: path do parquet de listings do snapshot mais antigo
        - calendar_earlier: path do parquet de calendário do snapshot mais antigo
        - date_earlier: data do scrape mais antigo (str YYYY-MM-DD)
        - listings_later: path do parquet de listings do snapshot mais recente
        - calendar_later: path do parquet de calendário do snapshot mais recente
        - date_later: data do scrape mais recente (str YYYY-MM-DD)

    Default (backward-compatible): usa apenas Jun/Set 2025.
    Com múltiplos pares: weighted average de elasticidade (pares recentes > peso).
    """

    def __init__(self, snapshot_pairs: list[dict] | None = None):
        if snapshot_pairs is None:
            self.snapshot_pairs = [
                {
                    "listings_earlier": str(PROCESSED_DATA_PATH / "listings_jun2025.parquet"),
                    "calendar_earlier": str(PROCESSED_DATA_PATH / "calendar_jun2025.parquet"),
                    "date_earlier": "2025-06-24",
                    "listings_later": str(PROCESSED_DATA_PATH / "listings_set2025.parquet"),
                    "calendar_later": str(PROCESSED_DATA_PATH / "calendar_set2025.parquet"),
                    "date_later": "2025-09-26",
                }
            ]
        else:
            self.snapshot_pairs = snapshot_pairs

    @staticmethod
    def _load_prices(path: str) -> pd.DataFrame:
        df = pd.read_parquet(path, columns=["id", "price", "neighbourhood_cleansed", "room_type"])
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
        return df.dropna(subset=["price"]).rename(columns={"id": "listing_id"})

    def _panel_b_estimates(self, pair: dict) -> pd.DataFrame:
        """
        Estima elasticidade b por segmento (neighbourhood, room_type) para um par de snapshots.
        Retorna DataFrame com colunas: neighbourhood_cleansed, room_type, b, n.
        """
        occ_early = _occupancy(Path(pair["calendar_earlier"]), pd.Timestamp(pair["date_earlier"]))
        occ_late = _occupancy(Path(pair["calendar_later"]), pd.Timestamp(pair["date_later"]))

        prices_early = self._load_prices(pair["listings_earlier"])
        prices_late = self._load_prices(pair["listings_later"])

        panel = (
            prices_early[["listing_id", "price", "neighbourhood_cleansed", "room_type"]]
            .merge(prices_late[["listing_id", "price"]], on="listing_id", suffixes=("_early", "_late"))
            .merge(occ_early.rename("occ_early"), on="listing_id", how="inner")
            .merge(occ_late.rename("occ_late"), on="listing_id", how="inner")
        )
        panel = panel[(panel["price_early"] > 0) & (panel["price_late"] > 0)]
        panel = panel[(panel["occ_early"] >= MIN_OCCUPANCY) | (panel["occ_late"] >= MIN_OCCUPANCY)]
        panel["delta_log_price"] = np.log(panel["price_late"]) - np.log(panel["price_early"])
        panel["delta_logit_occ"] = _safe_logit(panel["occ_late"]) - _safe_logit(panel["occ_early"])
        panel_varying = panel[panel["delta_log_price"].abs() > 0.01].copy()

        rows = []
        for (neighbourhood, room_type), seg in panel_varying.groupby(
            ["neighbourhood_cleansed", "room_type"]
        ):
            if len(seg) < MIN_SEGMENT_PANEL:
                continue
            dm_dp = seg["delta_log_price"] - seg["delta_log_price"].mean()
            dm_do = seg["delta_logit_occ"] - seg["delta_logit_occ"].mean()
            if dm_dp.std() <= 1e-4:
                continue
            try:
                lr = LinearRegression(fit_intercept=False)
                lr.fit(dm_dp.values.reshape(-1, 1), dm_do.values)
                b_val = float(lr.coef_[0])
                if b_val < 0:
                    rows.append({
                        "neighbourhood_cleansed": neighbourhood,
                        "room_type": room_type,
                        "b": b_val,
                        "n": len(seg),
                    })
            except Exception:
                pass

        return pd.DataFrame(rows)

    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # ── 1. Coletar estimativas de b para todos os pares disponíveis ───────
        all_b_estimates: list[pd.DataFrame] = []
        for pair in self.snapshot_pairs:
            estimates = self._panel_b_estimates(pair)
            if not estimates.empty:
                all_b_estimates.append(estimates)
                logger.info(
                    f"Par {pair['date_earlier']}→{pair['date_later']}: "
                    f"{len(estimates)} segmentos com elasticidade estimada"
                )

        # Weighted average: pares mais recentes têm peso 2× maior que o anterior
        # w_i = 2^i / sum(2^j) onde i=0 é o mais antigo
        combined_b: dict[tuple, float] = {}  # (neighbourhood, room_type) → b ponderado
        if all_b_estimates:
            n_pairs = len(all_b_estimates)
            weights = np.array([2.0 ** i for i in range(n_pairs)])
            weights /= weights.sum()

            for i, (est, w) in enumerate(zip(all_b_estimates, weights)):
                for _, row in est.iterrows():
                    key = (row["neighbourhood_cleansed"], row["room_type"])
                    if key not in combined_b:
                        combined_b[key] = {"b_sum": 0.0, "w_sum": 0.0}
                    combined_b[key]["b_sum"] += w * row["b"] * row["n"]
                    combined_b[key]["w_sum"] += w * row["n"]

        # Elasticidade ponderada final por segmento
        panel_b_final: dict[tuple, float] = {
            k: v["b_sum"] / v["w_sum"]
            for k, v in combined_b.items()
            if v["w_sum"] > 0
        }
        logger.info(f"Elasticidade ponderada calculada para {len(panel_b_final)} segmentos")

        # ── 2. Dados do snapshot mais recente (para a, cross-sectional, features) ──
        latest_pair = self.snapshot_pairs[-1]
        occ_latest = _occupancy(
            Path(latest_pair["calendar_later"]),
            pd.Timestamp(latest_pair["date_later"])
        )
        prices_latest = self._load_prices(latest_pair["listings_later"])
        logger.info(f"Snapshot mais recente: {latest_pair['date_later']} | {len(prices_latest):,} listings")

        # ── 3. Competição local (para price_premium e preço ótimo) ───────────
        comp = pd.read_parquet(
            PROCESSED_DATA_PATH / "competition_features.parquet",
            columns=["comp_price_p50", "comp_price_p75"],
        )

        # Dataset cross-sectional (snapshot mais recente)
        prices_latest_active = prices_latest.merge(
            occ_latest.rename("occ_latest"), on="listing_id", how="inner"
        )
        prices_latest_active = prices_latest_active[
            prices_latest_active["occ_latest"] >= MIN_OCCUPANCY
        ]
        prices_latest_active = prices_latest_active.merge(
            comp, left_on="listing_id", right_index=True, how="left"
        )
        prices_latest_active["price_premium"] = (
            prices_latest_active["price"] / prices_latest_active["comp_price_p50"].clip(lower=1)
        )

        # ── 4. Estimação da curva de demanda por segmento ────────────────────
        demand_params: dict = {}

        for (neighbourhood, room_type), seg_cs in prices_latest_active.groupby(
            ["neighbourhood_cleansed", "room_type"]
        ):
            comp_p50 = seg_cs["comp_price_p50"].median()
            n_cs = len(seg_cs)

            # Tentativa 1: elasticidade ponderada dos pares de snapshots ──────
            b = panel_b_final.get((neighbourhood, room_type))
            a = None

            if b is not None and b < 0:
                # Intercept: nível médio de logit(occupancy) no snapshot mais recente
                mean_occ = seg_cs["occ_latest"].clip(MIN_OCCUPANCY, 1 - MIN_OCCUPANCY).mean()
                mean_pp = np.log(seg_cs["price_premium"].clip(0.1, 10).mean())
                a = _safe_logit(np.array([mean_occ]))[0] - b * mean_pp
                method = "panel"
            else:
                b = None  # resetar para tentar cross-sectional

            # Tentativa 2: cross-sectional (fallback) ─────────────────────────
            if b is None and n_cs >= MIN_SEGMENT_CROSS:
                X = np.log(seg_cs["price_premium"].clip(0.1, 10).values).reshape(-1, 1)
                y = _safe_logit(seg_cs["occ_latest"].values)
                try:
                    lr = LinearRegression()
                    lr.fit(X, y)
                    b_cs = float(lr.coef_[0])
                    if b_cs < 0:
                        b = b_cs
                        a = float(lr.intercept_)
                        method = "cross"
                except Exception:
                    pass

            # Fallback final: elasticidade neutra → preço ótimo = mediana
            if b is None or b >= 0:
                demand_params[(neighbourhood, room_type)] = {
                    "a": None, "b": None, "n": n_cs,
                    "comp_p50": float(comp_p50),
                    "optimal_multiplier": 1.0,
                    "method": "fallback",
                }
                continue

            comp_p75 = seg_cs["comp_price_p75"].median()
            _, _, strategy = _optimal_price(a, b, comp_p50, comp_p75)
            opt_mult = (
                min(float(PRICE_GRID[np.argmax(
                    PRICE_GRID * comp_p50 * expit(a + b * np.log(PRICE_GRID))
                )]), MAX_OPTIMAL_MULT)
                if b < -1
                else float(comp_p75 / comp_p50) if comp_p50 > 0 else 1.25
            )

            demand_params[(neighbourhood, room_type)] = {
                "a": float(a), "b": float(b), "n": n_cs,
                "comp_p50": float(comp_p50),
                "comp_p75": float(comp_p75),
                "optimal_multiplier": float(opt_mult),
                "method": method,
                "strategy": strategy,
            }

        # ── 5. Features por listing ───────────────────────────────────────────
        rows = []
        for _, row in prices_latest_active.iterrows():
            lid = row["listing_id"]
            nbh = row["neighbourhood_cleansed"]
            rt = row["room_type"]
            occ = row["occ_latest"]
            pp = row["price_premium"]
            cp50 = row["comp_price_p50"]

            params = demand_params.get((nbh, rt), {})
            a_p = params.get("a")
            b_p = params.get("b")

            cp75 = params.get("comp_p75", cp50 * 1.25)
            if a_p is not None and b_p is not None:
                opt_p, opt_o, _ = _optimal_price(a_p, b_p, cp50, cp75)
            else:
                opt_p = float(cp50) if pd.notna(cp50) else row["price"]
                opt_o = float(occ)

            rows.append({
                "listing_id": lid,
                "occupancy_rate": float(occ),
                "price_premium": float(pp),
                "revenue_optimal_price": opt_p,
                "expected_occupancy_at_optimal": opt_o,
            })

        features = pd.DataFrame(rows).set_index("listing_id")

        output_path = PROCESSED_DATA_PATH / "demand_features.parquet"
        features.to_parquet(output_path)

        # Resumo
        methods = {v["method"]: 0 for v in demand_params.values()}
        for v in demand_params.values():
            methods[v["method"]] += 1

        logger.info(f"Demand features — shape: {features.shape}")
        logger.info(f"Segmentos por método: {methods}")
        mults = [v["optimal_multiplier"] for v in demand_params.values() if v["method"] != "fallback"]
        if mults:
            logger.info(
                f"Multiplicador ótimo — mediana: {np.median(mults):.2f} "
                f"| P25: {np.percentile(mults, 25):.2f} "
                f"| P75: {np.percentile(mults, 75):.2f}"
            )

        params_path = PROCESSED_DATA_PATH / "demand_params.joblib"
        joblib.dump(demand_params, params_path)
        logger.info(f"Demand params saved — {len(demand_params)} segmentos")

        return str(output_path)

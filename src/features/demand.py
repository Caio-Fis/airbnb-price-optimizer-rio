"""
Módulo de otimização de receita via curva de demanda.

Abordagem: elasticidade de preço estimada por painel Jun→Set 2025.
  - Para cada listing com dados nos dois snapshots: Δlogit(occ) ~ b * Δlog(price)
  - Controla efeitos fixos por listing (qualidade, localização) e sazonalidade
  - Segmentos sem variação suficiente usam fallback cross-sectional

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
    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # ── 1. Ocupação por snapshot ──────────────────────────────────────────
        occ_jun = _occupancy(PROCESSED_DATA_PATH / "calendar_jun2025.parquet", JUN_SCRAPE)
        occ_set = _occupancy(PROCESSED_DATA_PATH / "calendar_set2025.parquet", SET_SCRAPE)
        logger.info(f"Ocupação — jun: {len(occ_jun):,} | set: {len(occ_set):,}")

        # ── 2. Preços de ambos os snapshots ──────────────────────────────────
        def _load_prices(fname):
            df = pd.read_parquet(
                PROCESSED_DATA_PATH / fname,
                columns=["id", "price", "neighbourhood_cleansed", "room_type"],
            )
            df["price"] = pd.to_numeric(
                df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )
            return df.dropna(subset=["price"]).rename(columns={"id": "listing_id"})

        prices_jun = _load_prices("listings_jun2025.parquet")
        prices_set = _load_prices("listings_set2025.parquet")

        # ── 3. Dataset de painel (Jun ∩ Set) ──────────────────────────────────
        panel = (
            prices_jun[["listing_id", "price", "neighbourhood_cleansed", "room_type"]]
            .merge(prices_set[["listing_id", "price"]], on="listing_id", suffixes=("_jun", "_set"))
            .merge(occ_jun.rename("occ_jun"), on="listing_id", how="inner")
            .merge(occ_set.rename("occ_set"), on="listing_id", how="inner")
        )
        panel = panel[(panel["price_jun"] > 0) & (panel["price_set"] > 0)]
        panel = panel[(panel["occ_jun"] >= MIN_OCCUPANCY) | (panel["occ_set"] >= MIN_OCCUPANCY)]

        # Variações dentro do listing
        panel["delta_log_price"] = np.log(panel["price_set"]) - np.log(panel["price_jun"])
        panel["delta_logit_occ"] = _safe_logit(panel["occ_set"]) - _safe_logit(panel["occ_jun"])

        # Remover listings sem variação de preço (inúteis para painel)
        panel_varying = panel[panel["delta_log_price"].abs() > 0.01].copy()
        logger.info(f"Painel: {len(panel_varying):,} listings com variação de preço")

        # ── 4. Competição local (para price_premium e preço ótimo) ───────────
        comp = pd.read_parquet(
            PROCESSED_DATA_PATH / "competition_features.parquet",
            columns=["comp_price_p50", "comp_price_p75"],
        )

        # Dataset cross-sectional (set snapshot — mais recente)
        prices_set_active = prices_set.merge(
            occ_set.rename("occ_set"), on="listing_id", how="inner"
        )
        prices_set_active = prices_set_active[prices_set_active["occ_set"] >= MIN_OCCUPANCY]
        prices_set_active = prices_set_active.merge(
            comp, left_on="listing_id", right_index=True, how="left"
        )
        prices_set_active["price_premium"] = (
            prices_set_active["price"] / prices_set_active["comp_price_p50"].clip(lower=1)
        )

        # ── 5. Estimação da curva de demanda por segmento ────────────────────
        demand_params: dict = {}

        for (neighbourhood, room_type), seg_cs in prices_set_active.groupby(
            ["neighbourhood_cleansed", "room_type"]
        ):
            comp_p50 = seg_cs["comp_price_p50"].median()
            n_cs = len(seg_cs)

            # Tentativa 1: painel (within-listing elasticity) ─────────────────
            seg_panel = panel_varying[
                (panel_varying["neighbourhood_cleansed"] == neighbourhood)
                & (panel_varying["room_type"] == room_type)
            ]

            a, b = None, None

            if len(seg_panel) >= MIN_SEGMENT_PANEL:
                # Demeaning por segmento para remover sazonalidade
                dm_dp = seg_panel["delta_log_price"] - seg_panel["delta_log_price"].mean()
                dm_do = seg_panel["delta_logit_occ"] - seg_panel["delta_logit_occ"].mean()

                if dm_dp.std() > 1e-4:
                    try:
                        lr = LinearRegression(fit_intercept=False)
                        lr.fit(dm_dp.values.reshape(-1, 1), dm_do.values)
                        b_panel = float(lr.coef_[0])
                        if b_panel < 0:  # elasticidade negativa — válida
                            b = b_panel
                            # Intercept: nível médio de logit(occupancy)
                            mean_occ = seg_cs["occ_set"].clip(MIN_OCCUPANCY, 1 - MIN_OCCUPANCY).mean()
                            mean_pp = np.log(seg_cs["price_premium"].clip(0.1, 10).mean())
                            a = _safe_logit(np.array([mean_occ]))[0] - b * mean_pp
                    except Exception:
                        pass

            # Tentativa 2: cross-sectional (fallback) ─────────────────────────
            if b is None and n_cs >= MIN_SEGMENT_CROSS:
                X = np.log(seg_cs["price_premium"].clip(0.1, 10).values).reshape(-1, 1)
                y = _safe_logit(seg_cs["occ_set"].values)
                try:
                    lr = LinearRegression()
                    lr.fit(X, y)
                    b_cs = float(lr.coef_[0])
                    if b_cs < 0:
                        b = b_cs
                        a = float(lr.intercept_)
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

            method = "panel" if len(seg_panel) >= MIN_SEGMENT_PANEL else "cross"
            demand_params[(neighbourhood, room_type)] = {
                "a": float(a), "b": float(b), "n": n_cs,
                "comp_p50": float(comp_p50),
                "comp_p75": float(comp_p75),
                "optimal_multiplier": float(opt_mult),
                "method": method,
                "strategy": strategy,
            }

        # ── 6. Features por listing ───────────────────────────────────────────
        rows = []
        for _, row in prices_set_active.iterrows():
            lid = row["listing_id"]
            nbh = row["neighbourhood_cleansed"]
            rt = row["room_type"]
            occ = row["occ_set"]
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

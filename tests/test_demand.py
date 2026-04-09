"""Testes para o módulo de estimação de demanda."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestOccupancy:
    def test_occupancy_basic(self, tmp_path):
        from src.features.demand import _occupancy

        scrape = pd.Timestamp("2025-09-26")
        dates = pd.date_range(scrape, periods=10, freq="D")
        df = pd.DataFrame({
            "listing_id": [1] * 10,
            "date": dates,
            "available": ["t", "f"] * 5,  # 50% disponível → ocupação = 0.5
        })
        path = tmp_path / "calendar.parquet"
        df.to_parquet(path)

        result = _occupancy(path, scrape)
        assert 1 in result.index
        assert result[1] == pytest.approx(0.5, abs=0.01)

    def test_occupancy_fully_booked(self, tmp_path):
        from src.features.demand import _occupancy

        scrape = pd.Timestamp("2025-09-26")
        dates = pd.date_range(scrape, periods=5, freq="D")
        df = pd.DataFrame({
            "listing_id": [2] * 5,
            "date": dates,
            "available": ["f"] * 5,  # 100% ocupado
        })
        path = tmp_path / "calendar.parquet"
        df.to_parquet(path)

        result = _occupancy(path, scrape)
        assert result[2] == pytest.approx(1.0, abs=0.01)

    def test_occupancy_excludes_out_of_window(self, tmp_path):
        from src.features.demand import _occupancy

        scrape = pd.Timestamp("2025-09-26")
        # Datas antes do scrape — não devem contar
        dates_before = pd.date_range("2025-01-01", periods=10, freq="D")
        dates_after = pd.date_range(scrape, periods=5, freq="D")
        df = pd.DataFrame({
            "listing_id": [3] * 15,
            "date": list(dates_before) + list(dates_after),
            "available": ["t"] * 10 + ["f"] * 5,
        })
        path = tmp_path / "calendar.parquet"
        df.to_parquet(path)

        result = _occupancy(path, scrape)
        # Apenas as 5 datas dentro do window contam — todas "f" → ocupação = 1.0
        assert result[3] == pytest.approx(1.0, abs=0.01)


class TestSafeLogit:
    def test_clips_extremes(self):
        from src.features.demand import _safe_logit

        out = _safe_logit(np.array([0.0, 1.0, 0.5]))
        assert np.all(np.isfinite(out)), "Não deve produzir inf/nan nos extremos"
        assert out[2] == pytest.approx(0.0, abs=1e-5)  # logit(0.5) = 0

    def test_monotone(self):
        from src.features.demand import _safe_logit

        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        out = _safe_logit(x)
        assert np.all(np.diff(out) > 0), "logit deve ser monotonicamente crescente"


class TestOptimalPrice:
    def test_elastic_demand(self):
        """b < -1: deve encontrar máximo interior (revenue_optimal)."""
        from src.features.demand import _optimal_price

        opt_price, opt_occ, strategy = _optimal_price(
            a=0.5, b=-2.0, comp_p50=200.0, comp_p75=250.0
        )
        assert strategy == "revenue_optimal"
        assert opt_price > 0
        assert 0.0 < opt_occ <= 1.0

    def test_inelastic_demand(self):
        """b >= -1: sem máximo interior → premium_positioning no P75."""
        from src.features.demand import _optimal_price

        opt_price, opt_occ, strategy = _optimal_price(
            a=0.5, b=-0.5, comp_p50=200.0, comp_p75=250.0
        )
        assert strategy == "premium_positioning"
        assert opt_price == pytest.approx(250.0, abs=1e-2)

    def test_elastic_price_bounded(self):
        """Preço ótimo não deve exceder MAX_OPTIMAL_MULT × comp_p50."""
        from src.features.demand import _optimal_price, MAX_OPTIMAL_MULT

        opt_price, _, _ = _optimal_price(
            a=5.0, b=-1.5, comp_p50=100.0, comp_p75=150.0
        )
        assert opt_price <= MAX_OPTIMAL_MULT * 100.0


class TestDemandPipeline:
    def _make_listings(self, n=30) -> pd.DataFrame:
        np.random.seed(42)
        prices = np.random.uniform(100, 500, n)
        return pd.DataFrame({
            "id": range(n),
            "price": [f"${p:.2f}" for p in prices],
            "neighbourhood_cleansed": ["Copacabana"] * (n // 2) + ["Ipanema"] * (n - n // 2),
            "room_type": ["Entire home/apt"] * n,
        })

    def _make_calendar(self, listing_ids, scrape_date, n_days=90) -> pd.DataFrame:
        rows = []
        for lid in listing_ids:
            dates = pd.date_range(scrape_date, periods=n_days, freq="D")
            avail = np.random.choice(["t", "f"], size=n_days)
            rows.append(pd.DataFrame({
                "listing_id": lid,
                "date": dates,
                "available": avail,
            }))
        return pd.concat(rows, ignore_index=True)

    def _make_competition(self, listing_ids) -> pd.DataFrame:
        return pd.DataFrame({
            "comp_price_p50": [250.0] * len(listing_ids),
            "comp_price_p75": [350.0] * len(listing_ids),
        }, index=listing_ids)

    def test_run_produces_output(self, tmp_path):
        from src.features.demand import DemandPipeline

        listings_jun = self._make_listings(30)
        listings_set = self._make_listings(30)
        ids = list(range(30))
        cal_jun = self._make_calendar(ids, "2025-06-24")
        cal_set = self._make_calendar(ids, "2025-09-26")
        competition = self._make_competition(ids)

        def mock_read_parquet(path, **kwargs):
            path = str(path)
            cols = kwargs.get("columns")
            if "listings_jun" in path:
                return listings_jun[cols] if cols else listings_jun
            elif "listings_set" in path:
                return listings_set[cols] if cols else listings_set
            elif "calendar_jun" in path:
                return cal_jun[cols] if cols else cal_jun
            elif "calendar_set" in path:
                return cal_set[cols] if cols else cal_set
            elif "competition" in path:
                return competition[cols] if cols else competition
            return pd.DataFrame()

        with patch("src.features.demand.PROCESSED_DATA_PATH", tmp_path), \
             patch("pandas.read_parquet", side_effect=mock_read_parquet):
            pipeline = DemandPipeline()
            output_path = pipeline.run()

        assert Path(output_path).exists()
        result = pd.read_parquet(output_path)
        assert "revenue_optimal_price" in result.columns
        assert "occupancy_rate" in result.columns
        assert (tmp_path / "demand_params.joblib").exists()

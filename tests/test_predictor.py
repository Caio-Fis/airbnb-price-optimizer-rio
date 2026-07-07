"""Testes para a classe Predictor (inferência)."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import date


@pytest.fixture
def mock_predictor():
    from src.serving.predict import Predictor

    with patch.object(Predictor, "_load"):
        p = Predictor()

    p.model = MagicMock()
    p.model.predict.return_value = np.array([np.log1p(350.0)])

    target_enc = MagicMock()
    target_enc.transform.return_value = pd.DataFrame({"neighbourhood_enc": [5.8]})

    mlb = MagicMock()
    mlb.transform.return_value = np.zeros((1, 16))

    p.encoders = {
        "target_encoder": target_enc,
        "mlb": mlb,
        "price_p99": 2000.0,
    }
    p.price_p99 = 2000.0
    p.feature_names = None
    p.competition_stats = {
        ("Copacabana", "Entire home/apt"): {
            "count": 120, "p25": 200.0, "p50": 350.0, "p75": 500.0
        }
    }
    p.demand_params = {}
    p.seasonal_by_dow = {i: 0.0 for i in range(7)}
    p.metro_stations = []
    p.training_stats = {"days_since_last_review_median": 45.0}
    p.prediction_intervals = {"p10_pct": -0.30, "p90_pct": 0.35}
    p.model_version = "test"
    return p


def make_request(**kwargs):
    from src.serving.schemas import PredictionRequest

    defaults = dict(
        neighbourhood="Copacabana",
        room_type="Entire home/apt",
        accommodates=2,
        bathrooms=1.0,
        bedrooms=1,
        beds=1,
    )
    defaults.update(kwargs)
    return PredictionRequest(**defaults)


class TestRankFromPercentiles:
    def test_at_p25(self):
        from src.serving.predict import _rank_from_percentiles

        rank = _rank_from_percentiles(200.0, 200.0, 350.0, 500.0)
        assert rank == pytest.approx(0.25, abs=0.05)

    def test_at_p50(self):
        from src.serving.predict import _rank_from_percentiles

        rank = _rank_from_percentiles(350.0, 200.0, 350.0, 500.0)
        assert rank == pytest.approx(0.50, abs=0.05)

    def test_at_p75(self):
        from src.serving.predict import _rank_from_percentiles

        rank = _rank_from_percentiles(500.0, 200.0, 350.0, 500.0)
        assert rank == pytest.approx(0.75, abs=0.05)

    def test_below_p25(self):
        from src.serving.predict import _rank_from_percentiles

        rank = _rank_from_percentiles(50.0, 200.0, 350.0, 500.0)
        assert rank < 0.25

    def test_above_p75(self):
        from src.serving.predict import _rank_from_percentiles

        rank = _rank_from_percentiles(1000.0, 200.0, 350.0, 500.0)
        assert rank > 0.75
        assert rank <= 1.0

    def test_invalid_percentiles_returns_half(self):
        from src.serving.predict import _rank_from_percentiles

        rank = _rank_from_percentiles(300.0, 0.0, 0.0, 500.0)
        assert rank == 0.5


class TestGeoFeatures:
    def test_no_latlon_returns_zeros(self, mock_predictor):
        result = mock_predictor._geo_features(None, None)
        assert all(v == 0.0 for v in result.values())

    def test_with_latlon_returns_distances(self, mock_predictor):
        # Coordenadas de Copacabana
        result = mock_predictor._geo_features(-22.971, -43.182)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result.values())
        assert all(v >= 0.0 for v in result.values())


class TestCompetitionFeatures:
    def test_known_group(self, mock_predictor):
        result = mock_predictor._competition_features("Copacabana", "Entire home/apt")
        assert result["comp_count"] == 120
        assert result["comp_price_p50"] == 350.0
        assert result["comp_price_rank"] == 0.5  # default no primeiro pass

    def test_unknown_group_returns_zeros(self, mock_predictor):
        result = mock_predictor._competition_features("Bairro Inexistente", "Entire home/apt")
        assert result["comp_count"] == 0.0
        assert result["comp_price_p50"] == 0.0

    def test_rank_override(self, mock_predictor):
        result = mock_predictor._competition_features(
            "Copacabana", "Entire home/apt", comp_price_rank_override=0.8
        )
        assert result["comp_price_rank"] == 0.8


class TestBuildFeatures:
    def test_shape_is_one_row(self, mock_predictor):
        req = make_request()
        df = mock_predictor._build_features(req)
        assert df.shape[0] == 1

    def test_rank_override_propagates(self, mock_predictor):
        req = make_request()
        df = mock_predictor._build_features(req, comp_price_rank_override=0.9)
        assert df["comp_price_rank"].iloc[0] == 0.9


class TestRevenueOptimal:
    def test_fallback_when_no_params(self, mock_predictor):
        price, occ, strategy = mock_predictor._revenue_optimal("Bairro Sem Dados", "Entire home/apt")
        assert price is None
        assert occ is None
        assert strategy == "fallback"

    def test_elastic_demand(self, mock_predictor):
        mock_predictor.demand_params = {
            ("Copacabana", "Entire home/apt"): {
                "a": 0.5, "b": -2.0,
                "comp_p50": 350.0, "comp_p75": 450.0,
                "strategy": "revenue_optimal",
            }
        }
        price, occ, strategy = mock_predictor._revenue_optimal("Copacabana", "Entire home/apt")
        assert price > 0
        assert strategy == "revenue_optimal"
        assert 0.0 < occ <= 100.0


class TestTwoPassPrediction:
    def test_second_pass_uses_real_rank(self, mock_predictor):
        """Se o preço previsto cai acima do P50, o segundo pass deve usar rank > 0.5."""
        # model.predict sempre retorna log(351) ≈ log(350+1) → market_price ≈ 350
        # comp_p50 = 350 → rank deve ficar próximo a 0.5
        req = make_request()
        call_ranks = []

        original_build = mock_predictor._build_features

        def capture_build(r, comp_price_rank_override=None, listing_id=None):
            call_ranks.append(comp_price_rank_override)
            return original_build(r, comp_price_rank_override, listing_id)

        mock_predictor._build_features = capture_build
        mock_predictor.predict(req)

        # Primeiro call: None (0.5 default), segundo call: rank calculado
        assert call_ranks[0] is None
        assert call_ranks[1] is not None

    def test_prediction_uses_data_driven_intervals(self, mock_predictor):
        req = make_request()
        response = mock_predictor.predict(req)
        # Com p10_pct=-0.30, p90_pct=0.35 — os intervalos não devem ser ±15%
        ratio_low = response.price_range_low / response.predicted_price
        ratio_high = response.price_range_high / response.predicted_price
        assert ratio_low == pytest.approx(0.70, abs=0.02)
        assert ratio_high == pytest.approx(1.35, abs=0.02)

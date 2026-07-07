"""Testes para feature engineering."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestSeasonalityPipeline:
    def _make_series(self, n=365):
        import pandas as pd
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # Série com sazonalidade semanal sintética
        values = 200 + 30 * np.sin(2 * np.pi * np.arange(n) / 7) + np.random.normal(0, 5, n)
        return pd.Series(values, index=dates)

    def test_hw_features_shape(self):
        from src.features.tabular import SeasonalityPipeline
        pipeline = SeasonalityPipeline()
        series = self._make_series()
        features = pipeline._extract_hw_features(series)

        assert "hw_level" in features.columns
        assert "hw_trend" in features.columns
        assert "hw_seasonal" in features.columns
        assert "hw_residual" in features.columns
        assert "is_weekend" in features.columns
        assert len(features) == len(series)

    def test_hw_seasonal_amplitude(self):
        from src.features.tabular import SeasonalityPipeline
        pipeline = SeasonalityPipeline()
        series = self._make_series()
        features = pipeline._extract_hw_features(series)
        # Sazonalidade deve ter alguma variação
        assert features["hw_seasonal"].std() > 0

    def test_weekend_flag(self):
        from src.features.tabular import SeasonalityPipeline
        pipeline = SeasonalityPipeline()
        series = self._make_series()
        features = pipeline._extract_hw_features(series)
        # Deve haver fins de semana na amostra
        assert features["is_weekend"].sum() > 0


class TestTabularFeaturePipeline:
    def _make_df(self, n=100):
        np.random.seed(42)
        return pd.DataFrame({
            "id": range(n),
            "price": np.random.uniform(50, 500, n),
            "room_type": np.random.choice(["Entire home/apt", "Private room"], n),
            "neighbourhood_cleansed": np.random.choice(["Pinheiros", "Vila Madalena", "Centro"], n),
            "accommodates": np.random.randint(1, 8, n),
            "bathrooms": np.random.uniform(1, 3, n),
            "bedrooms": np.random.randint(1, 4, n),
            "beds": np.random.randint(1, 5, n),
            "minimum_nights": np.random.randint(1, 7, n),
            "maximum_nights": [365] * n,
            "number_of_reviews": np.random.randint(0, 100, n),
            "review_scores_rating": np.random.uniform(3, 5, n),
            "review_scores_cleanliness": np.random.uniform(3, 5, n),
            "review_scores_location": np.random.uniform(3, 5, n),
            "host_is_superhost": np.random.choice(["t", "f"], n),
            "instant_bookable": np.random.choice(["t", "f"], n),
            "calculated_host_listings_count": np.random.randint(1, 5, n),
            "availability_365": np.random.randint(0, 365, n),
            "amenities": ['["Wifi", "Kitchen", "Air conditioning"]'] * n,
            "cancellation_policy": ["flexible"] * n,
        })

    def test_price_cleaning(self):
        from src.features.tabular import TabularFeaturePipeline
        pipeline = TabularFeaturePipeline()
        df = self._make_df()
        df["price"] = df["price"].map(lambda x: f"${x:.2f}")
        result = pipeline._clean_price(df)
        assert result["price"].dtype == float
        assert "log_price" in result.columns
        assert (result["log_price"] > 0).all()

    def test_amenities_parsing(self):
        from src.features.tabular import TabularFeaturePipeline
        pipeline = TabularFeaturePipeline()
        df = self._make_df()
        result = pipeline._parse_amenities(df)
        assert "amenities_list" in result.columns
        assert isinstance(result["amenities_list"].iloc[0], list)

    def test_binary_encoding(self):
        from src.features.tabular import TabularFeaturePipeline
        pipeline = TabularFeaturePipeline()
        df = self._make_df()
        result = pipeline._encode_binary(df)
        assert result["instant_bookable"].isin([0, 1]).all()

    def test_no_nulls_in_numerics(self):
        from src.features.tabular import TabularFeaturePipeline
        pipeline = TabularFeaturePipeline()
        df = self._make_df()
        df.loc[0, "bathrooms"] = None
        result = pipeline._fill_numerics(df)
        assert result["bathrooms"].isna().sum() == 0


class TestPoiStats:
    def test_distance_and_count(self):
        from src.features.geo import poi_stats
        # POI 0.01° ao norte ≈ 1.112 km; outro a 0.001° ≈ 0.111 km
        pois = [(0.01, 0.0), (0.001, 0.0)]
        dist, count = poi_stats(np.array([0.0]), np.array([0.0]), pois, radius_km=0.5)
        assert dist[0] == pytest.approx(0.111, abs=0.005)
        assert count[0] == 1.0

    def test_count_within_radius(self):
        from src.features.geo import poi_stats
        # 3 POIs a ~111m e 1 a ~11 km
        pois = [(0.001, 0.0), (0.0, 0.001), (-0.001, 0.0), (0.1, 0.0)]
        _, count = poi_stats(np.array([0.0]), np.array([0.0]), pois, radius_km=0.5)
        assert count[0] == 3.0

    def test_vectorized_matches_single(self):
        from src.features.geo import poi_stats
        rng = np.random.default_rng(42)
        pois = [(-22.97 + rng.normal(0, 0.01), -43.19 + rng.normal(0, 0.01)) for _ in range(50)]
        lats = np.array([-22.971, -22.985])
        lons = np.array([-43.186, -43.20])
        dist_batch, count_batch = poi_stats(lats, lons, pois)
        for i in range(2):
            d, c = poi_stats(lats[i:i+1], lons[i:i+1], pois)
            assert d[0] == pytest.approx(dist_batch[i])
            assert c[0] == count_batch[i]


class TestGeoServingParity:
    """As features geo do serving devem bater com as do pipeline offline."""

    @pytest.mark.skipif(
        not __import__("pathlib").Path("data/processed/geo_features.parquet").exists(),
        reason="artefatos locais ausentes",
    )
    def test_offline_matches_serving(self):
        from pathlib import Path
        from src.serving.predict import Predictor

        gf = pd.read_parquet("data/processed/geo_features.parquet")
        listings = pd.read_parquet(
            "data/processed/listings_slim.parquet",
            columns=["id", "latitude", "longitude"],
        ).set_index("id")

        predictor = Predictor()
        listing_id = gf.index[100]
        row = listings.loc[listing_id]
        serving = predictor._geo_features(float(row["latitude"]), float(row["longitude"]))

        offline = gf.loc[listing_id]
        common = [c for c in gf.columns if c in serving]
        assert len(common) >= 12
        for col in common:
            assert serving[col] == pytest.approx(offline[col], abs=0.01), col


class TestBairroFeatures:
    def test_normalize(self):
        from src.features.bairro import normalize_bairro
        assert normalize_bairro("São Conrado") == "sao conrado"
        assert normalize_bairro("  Barra   da Tijuca ") == "barra da tijuca"

    @pytest.mark.skipif(
        not __import__("pathlib").Path("data/external/ips_bairros_2022.csv").exists(),
        reason="CSV do IPS ausente",
    )
    def test_lookup_and_override(self):
        from src.features.bairro import load_ips_lookup
        lookup = load_ips_lookup()
        assert len(lookup) > 150
        assert lookup["leblon"]["ips_2022"] > lookup["rocinha"]["ips_2022"]
        # override Gericinó → Bangu
        assert lookup["gericino"] == lookup["bangu"]

    @pytest.mark.skipif(
        not __import__("pathlib").Path("data/external/ips_bairros_2022.csv").exists(),
        reason="CSV do IPS ausente",
    )
    def test_serving_fallback_unknown_bairro(self):
        from unittest.mock import patch
        with patch("src.serving.predict.Predictor._load"):
            from src.serving.predict import Predictor
            from src.features.bairro import load_ips_lookup, IPS_FEATURES
            import numpy as np
            p = Predictor()
            p.ips_lookup = load_ips_lookup()
            p.ips_medians = {f: 50.0 for f in IPS_FEATURES}

            known = p._bairro_features("Copacabana")
            assert known["bairro_ips_missing"] == 0
            assert known["bairro_ips_2022"] > 0

            unknown = p._bairro_features("Bairro Inexistente")
            assert unknown["bairro_ips_missing"] == 1
            assert unknown["bairro_ips_2022"] == 50.0


class TestSeasonalMultiplier:
    def _predictor(self):
        from unittest.mock import patch
        with patch("src.serving.predict.Predictor._load"):
            from src.serving.predict import Predictor
            p = Predictor()
        p.seasonal_factors = {
            "dow": {0: 0.98, 1: 0.97, 2: 0.99, 3: 1.0, 4: 1.02, 5: 1.02, 6: 1.01},
            "events": {"reveillon": 1.60, "carnaval": 1.34, "feriado": 1.0},
            "dampening": 0.5,
            "clip": [0.8, 2.5],
        }
        return p

    def test_no_date_returns_one(self):
        mult, note = self._predictor()._seasonal_multiplier(None)
        assert mult == 1.0 and note is None

    def test_carnaval_above_regular_day(self):
        from datetime import date
        p = self._predictor()
        carnaval, note_c = p._seasonal_multiplier(date(2026, 2, 16))  # segunda de Carnaval
        regular, _ = p._seasonal_multiplier(date(2026, 5, 12))  # terça comum
        assert carnaval > 1.1
        assert 0.95 < regular < 1.05
        assert "Carnaval" in note_c

    def test_reveillon(self):
        from datetime import date
        p = self._predictor()
        mult, note = p._seasonal_multiplier(date(2026, 12, 31))
        assert mult > 1.2
        assert "Réveillon" in note

    def test_clip_bounds(self):
        from datetime import date
        p = self._predictor()
        p.seasonal_factors["events"]["reveillon"] = 100.0
        mult, _ = p._seasonal_multiplier(date(2026, 12, 31))
        assert mult == 2.5

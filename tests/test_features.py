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

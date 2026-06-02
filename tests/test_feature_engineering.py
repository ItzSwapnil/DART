"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from ml.trading_ai_v3 import FeatureEngineer


@pytest.fixture
def dummy_ohlcv_data() -> pd.DataFrame:
    """Generate dummy OHLCV data for testing."""
    np.random.seed(42)
    n_points = 100
    dates = pd.date_range(start="2026-05-01", periods=n_points, freq="min")

    # Simulate a random walk for prices
    close = 100.0 + np.cumsum(np.random.normal(0, 0.5, n_points))
    open_p = close + np.random.normal(0, 0.2, n_points)
    high = np.maximum(open_p, close) + np.random.exponential(0.1, n_points)
    low = np.minimum(open_p, close) - np.random.exponential(0.1, n_points)
    volume = np.random.randint(100, 1000, n_points)

    return pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_engineer_features_shape(dummy_ohlcv_data: pd.DataFrame) -> None:
    """Test that engineered features DataFrame has correct shape and no NaNs."""
    features = FeatureEngineer.engineer_features(dummy_ohlcv_data, n_lags=5)

    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    # NaNs should be filled and dropped
    assert not features.isna().any().any()

    # Check that expected features are present
    assert "returns" in features.columns
    assert "volatility" in features.columns
    assert "rsi_10" in features.columns
    assert "macd_10" in features.columns
    assert "atr" in features.columns
    assert "bb_upper" in features.columns
    assert "bb_squeeze" in features.columns
    assert "return_lag_1" in features.columns
    assert "volume_change" in features.columns


def test_engineer_features_insufficient_data() -> None:
    """Test that feature engineering returns empty DataFrame for insufficient data."""
    df = pd.DataFrame({"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [100]})
    features = FeatureEngineer.engineer_features(df)
    # With only 1 row, rolling calculations can't be fulfilled (requires e.g. rolling 20)
    assert features.empty


def test_calculate_atr(dummy_ohlcv_data: pd.DataFrame) -> None:
    """Test ATR calculation helper."""
    atr = FeatureEngineer._calculate_atr(dummy_ohlcv_data, period=14)
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(dummy_ohlcv_data)
    # The first 13 rows should be NaN before rolling mean is satisfied
    assert atr.iloc[:13].isna().all()
    assert not atr.iloc[13:].isna().any()


def test_calculate_bollinger(dummy_ohlcv_data: pd.DataFrame) -> None:
    """Test Bollinger Bands calculation helper."""
    upper, lower = FeatureEngineer._calculate_bollinger(dummy_ohlcv_data, period=20, std_dev=2)
    assert isinstance(upper, pd.Series)
    assert isinstance(lower, pd.Series)
    assert (upper.dropna() >= lower.dropna()).all()

"""Unit tests for candlestick pattern recognition."""

import pandas as pd

from ml.candlestick_patterns import (
    detect_bearish_engulfing,
    detect_bullish_engulfing,
    detect_doji,
    detect_evening_star,
    detect_hammer,
    detect_inverted_hammer,
    detect_marubozu,
    detect_morning_star,
    get_all_patterns,
)


def test_detect_doji() -> None:
    """Test Doji detection logic."""
    # Create a Doji candle: open and close are equal
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [105.0, 105.0],
            "low": [95.0, 95.0],
            "close": [100.0, 101.5],  # 1st is Doji, 2nd is not
        }
    )
    doji = detect_doji(df)
    assert doji.iloc[0] == 1
    assert doji.iloc[1] == 0


def test_detect_hammer() -> None:
    """Test Hammer detection logic."""
    # Create a Hammer candle: small body, long lower shadow, small upper shadow
    # open=99.5, close=100, high=100.1, low=97.0
    # body = 0.5, lower_shadow = 2.5, upper_shadow = 0.1, rng = 3.1
    df = pd.DataFrame(
        {
            "open": [99.5],
            "high": [100.1],
            "low": [97.0],
            "close": [100.0],
        }
    )
    hammer = detect_hammer(df)
    assert hammer.iloc[0] == 1


def test_detect_inverted_hammer() -> None:
    """Test Inverted Hammer detection logic."""
    # Small body at bottom, long upper shadow, minimal lower shadow
    # open=97.5, close=97.0, high=100.0, low=96.9
    # body = 0.5, upper_shadow = 2.5, lower_shadow = 0.1, rng = 3.1
    df = pd.DataFrame(
        {
            "open": [97.5],
            "high": [100.0],
            "low": [96.9],
            "close": [97.0],
        }
    )
    inv_hammer = detect_inverted_hammer(df)
    assert inv_hammer.iloc[0] == 1


def test_detect_engulfing() -> None:
    """Test Bullish and Bearish Engulfing detection logic."""
    # Bullish engulfing: candle 1 is red, candle 2 is green engulfing candle 1
    df_bullish = pd.DataFrame(
        {
            "open": [101.0, 99.5],
            "high": [102.0, 102.0],
            "low": [99.0, 98.0],
            "close": [100.0, 101.5],
        }
    )
    bullish = detect_bullish_engulfing(df_bullish)
    assert bullish.iloc[0] == 0
    assert bullish.iloc[1] == 1

    # Bearish engulfing: candle 1 is green, candle 2 is red engulfing candle 1
    df_bearish = pd.DataFrame(
        {
            "open": [100.0, 101.5],
            "high": [102.0, 102.0],
            "low": [99.0, 98.0],
            "close": [101.0, 99.5],
        }
    )
    bearish = detect_bearish_engulfing(df_bearish)
    assert bearish.iloc[0] == 0
    assert bearish.iloc[1] == 1


def test_detect_marubozu() -> None:
    """Test Marubozu detection logic."""
    # Bullish Marubozu: huge body, virtually no shadows
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [110.0, 110.5],
            "low": [100.0, 99.5],
            "close": [110.0, 100.2],  # 1st is marubozu (+1), 2nd is not (0)
        }
    )
    marubozu = detect_marubozu(df)
    assert marubozu.iloc[0] == 1
    assert marubozu.iloc[1] == 0


def test_detect_stars() -> None:
    """Test Morning and Evening Star detection logic."""
    # Morning Star: Red -> Small body -> Green
    df_morning = pd.DataFrame(
        {
            "open": [105.0, 100.5, 100.0],
            "high": [106.0, 101.5, 104.0],
            "low": [99.0, 99.0, 99.5],
            "close": [100.0, 100.0, 103.5],
        }
    )
    morning = detect_morning_star(df_morning)
    assert morning.iloc[2] == 1

    # Evening Star: Green -> Small body -> Red
    df_evening = pd.DataFrame(
        {
            "open": [100.0, 104.5, 105.0],
            "high": [106.0, 106.0, 105.5],
            "low": [99.5, 103.5, 100.0],
            "close": [105.0, 105.0, 101.5],
        }
    )
    evening = detect_evening_star(df_evening)
    assert evening.iloc[2] == 1


def test_get_all_patterns() -> None:
    """Test get_all_patterns aggregator."""
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [105.0, 105.0],
            "Low": [95.0, 95.0],
            "Close": [100.0, 101.5],
        }
    )
    patterns = get_all_patterns(df)
    assert "pattern_doji" in patterns.columns
    assert "pattern_hammer" in patterns.columns
    assert "pattern_inverted_hammer" in patterns.columns
    assert "pattern_bullish_engulfing" in patterns.columns
    assert "pattern_bearish_engulfing" in patterns.columns
    assert "pattern_marubozu" in patterns.columns
    assert "pattern_morning_star" in patterns.columns
    assert "pattern_evening_star" in patterns.columns
    assert patterns["pattern_doji"].iloc[0] == 1

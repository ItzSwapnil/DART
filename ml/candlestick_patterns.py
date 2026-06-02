"""
Candlestick Pattern Recognition using pandas.
Implements dependency-free, vectorized candlestick pattern detection.
"""

import numpy as np
import pandas as pd


def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """
    Detect Doji pattern.
    Open and Close are virtually equal.
    """
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    # Avoid division by zero
    rng_safe = np.where(rng == 0, 1e-10, rng)
    is_doji = (body / rng_safe <= threshold) & (rng > 0)
    return pd.Series(is_doji.astype(int), index=df.index)


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Detect Hammer pattern.
    Small body at top, long lower shadow, minimal upper shadow.
    """
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    rng_safe = np.where(rng == 0, 1e-10, rng)

    lower_shadow = np.minimum(df["open"], df["close"]) - df["low"]
    upper_shadow = df["high"] - np.maximum(df["open"], df["close"])

    is_hammer = (rng > 0) & (lower_shadow >= 2 * body) & (upper_shadow <= 0.15 * rng_safe)
    return pd.Series(is_hammer.astype(int), index=df.index)


def detect_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Detect Inverted Hammer / Shooting Star pattern.
    Small body at bottom, long upper shadow, minimal lower shadow.
    """
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    rng_safe = np.where(rng == 0, 1e-10, rng)

    lower_shadow = np.minimum(df["open"], df["close"]) - df["low"]
    upper_shadow = df["high"] - np.maximum(df["open"], df["close"])

    is_inv_hammer = (rng > 0) & (upper_shadow >= 2 * body) & (lower_shadow <= 0.15 * rng_safe)
    return pd.Series(is_inv_hammer.astype(int), index=df.index)


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect Bullish Engulfing pattern.
    Green candle body completely engulfs previous red candle body.
    """
    if len(df) < 2:
        return pd.Series(0, index=df.index)

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    is_engulfing = (
        (df["close"] > df["open"])  # Current is green
        & (prev_close < prev_open)  # Previous was red
        & (df["open"] <= prev_close)  # Open <= previous close
        & (df["close"] >= prev_open)  # Close >= previous open
    )
    return pd.Series(is_engulfing.astype(int), index=df.index)


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect Bearish Engulfing pattern.
    Red candle body completely engulfs previous green candle body.
    """
    if len(df) < 2:
        return pd.Series(0, index=df.index)

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    is_engulfing = (
        (df["close"] < df["open"])  # Current is red
        & (prev_close > prev_open)  # Previous was green
        & (df["open"] >= prev_close)  # Open >= previous close
        & (df["close"] <= prev_open)  # Close <= previous open
    )
    return pd.Series(is_engulfing.astype(int), index=df.index)


def detect_marubozu(df: pd.DataFrame, threshold: float = 0.9) -> pd.Series:
    """
    Detect Marubozu pattern.
    A very large body with almost no shadow at either end.
    """
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    rng_safe = np.where(rng == 0, 1e-10, rng)

    is_marubozu = (body / rng_safe >= threshold) & (rng > 0)
    # Direction: +1 for bullish, -1 for bearish marubozu
    direction = np.where(df["close"] >= df["open"], 1, -1)
    return pd.Series((is_marubozu.astype(int) * direction), index=df.index)


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """
    Detect Morning Star trend reversal pattern (Bullish).
    Three-candle pattern: Red -> Small body star -> Green.
    """
    if len(df) < 3:
        return pd.Series(0, index=df.index)

    prev2_open = df["open"].shift(2)
    prev2_close = df["close"].shift(2)

    prev1_open = df["open"].shift(1)
    prev1_close = df["close"].shift(1)
    prev1_rng = df["high"].shift(1) - df["low"].shift(1)
    prev1_rng_safe = np.where(prev1_rng == 0, 1e-10, prev1_rng)

    is_star = (
        (prev2_close < prev2_open)  # 1st: Red
        & ((prev1_close - prev1_open).abs() / prev1_rng_safe <= 0.2)  # 2nd: Small body
        & (df["close"] > df["open"])  # 3rd: Green
        & (df["close"] > (prev2_open + prev2_close) / 2)  # Close above midpoint of 1st
    )
    return pd.Series(is_star.astype(int), index=df.index)


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """
    Detect Evening Star trend reversal pattern (Bearish).
    Three-candle pattern: Green -> Small body star -> Red.
    """
    if len(df) < 3:
        return pd.Series(0, index=df.index)

    prev2_open = df["open"].shift(2)
    prev2_close = df["close"].shift(2)

    prev1_open = df["open"].shift(1)
    prev1_close = df["close"].shift(1)
    prev1_rng = df["high"].shift(1) - df["low"].shift(1)
    prev1_rng_safe = np.where(prev1_rng == 0, 1e-10, prev1_rng)

    is_star = (
        (prev2_close > prev2_open)  # 1st: Green
        & ((prev1_close - prev1_open).abs() / prev1_rng_safe <= 0.2)  # 2nd: Small body
        & (df["close"] < df["open"])  # 3rd: Red
        & (df["close"] < (prev2_open + prev2_close) / 2)  # Close below midpoint of 1st
    )
    return pd.Series(is_star.astype(int), index=df.index)


def get_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all candlestick patterns and return a DataFrame.
    """
    # Standardize columns to lowercase
    cols_map = {col: col.lower() for col in df.columns}
    df_lower = df.rename(columns=cols_map)

    patterns = pd.DataFrame(index=df.index)
    patterns["pattern_doji"] = detect_doji(df_lower)
    patterns["pattern_hammer"] = detect_hammer(df_lower)
    patterns["pattern_inverted_hammer"] = detect_inverted_hammer(df_lower)
    patterns["pattern_bullish_engulfing"] = detect_bullish_engulfing(df_lower)
    patterns["pattern_bearish_engulfing"] = detect_bearish_engulfing(df_lower)
    patterns["pattern_marubozu"] = detect_marubozu(df_lower)
    patterns["pattern_morning_star"] = detect_morning_star(df_lower)
    patterns["pattern_evening_star"] = detect_evening_star(df_lower)

    return patterns

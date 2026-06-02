"""
Multi-modal Feature Extractor for DART
Implements comprehensive feature extraction from multiple data sources
as described in the project report.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

from ml.candlestick_patterns import get_all_patterns

logger = logging.getLogger("feature_extractor")


class TechnicalIndicators:
    """Advanced technical indicator calculations."""

    @staticmethod
    def calculate_rsi(prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicators."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bb_width = (upper_band - lower_band) / sma
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return upper_band, lower_band, bb_width, bb_position

    @staticmethod
    def calculate_stochastic(high, low, close, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

    @staticmethod
    def calculate_atr(high, low, close, window=14):
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    @staticmethod
    def calculate_adx(high, low, close, window=14):
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)

        atr = TechnicalIndicators.calculate_atr(high, low, close, window)
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def calculate_williams_r(high, low, close, window=14):
        """Calculate Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))

    @staticmethod
    def calculate_cci(high, low, close, window=20):
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def calculate_vwap(high, low, close, volume):
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def calculate_mfi(high, low, close, volume, window=14):
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)

        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()

        mf_ratio = positive_mf / negative_mf
        return 100 - (100 / (1 + mf_ratio))


class MarketStructureAnalyzer:
    """Analyze market microstructure patterns."""

    @staticmethod
    def calculate_price_patterns(ohlc_data):
        """Identify price patterns and structure."""
        patterns = {}

        # Support and resistance levels
        patterns["support_levels"] = MarketStructureAnalyzer._find_support_resistance(
            ohlc_data["low"],
            is_support=True,
        )
        patterns["resistance_levels"] = MarketStructureAnalyzer._find_support_resistance(
            ohlc_data["high"],
            is_support=False,
        )

        # Trend analysis
        patterns["trend_strength"] = MarketStructureAnalyzer._calculate_trend_strength(
            ohlc_data["close"],
        )
        patterns["trend_direction"] = MarketStructureAnalyzer._calculate_trend_direction(
            ohlc_data["close"],
        )

        # Volatility clustering
        patterns["volatility_regime"] = MarketStructureAnalyzer._identify_volatility_regime(
            ohlc_data,
        )

        # Gap analysis
        patterns["gaps"] = MarketStructureAnalyzer._identify_gaps(ohlc_data)

        return patterns

    @staticmethod
    def _find_support_resistance(price_series, window=20, is_support=True):
        """Find support and resistance levels."""
        if is_support:
            levels = price_series.rolling(window=window, center=True).min()
            condition = (
                (price_series == levels)
                & (price_series.shift(1) > price_series)
                & (price_series.shift(-1) > price_series)
            )
        else:
            levels = price_series.rolling(window=window, center=True).max()
            condition = (
                (price_series == levels)
                & (price_series.shift(1) < price_series)
                & (price_series.shift(-1) < price_series)
            )

        return price_series[condition].dropna()

    @staticmethod
    def _calculate_trend_strength(close_prices, window=20):
        """Calculate trend strength using linear regression."""

        def trend_strength(x):
            if len(x) < window:
                return 0
            y = np.arange(len(x))
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0

        return close_prices.rolling(window=window).apply(trend_strength)

    @staticmethod
    def _calculate_trend_direction(close_prices, short_window=10, long_window=30):
        """Calculate trend direction."""
        short_ma = close_prices.rolling(window=short_window).mean()
        long_ma = close_prices.rolling(window=long_window).mean()

        direction = np.where(short_ma > long_ma, 1, -1)
        return pd.Series(direction, index=close_prices.index)

    @staticmethod
    def _identify_volatility_regime(ohlc_data, window=20):
        """Identify volatility regime."""
        returns = ohlc_data["close"].pct_change()
        rolling_vol = returns.rolling(window=window).std()
        vol_percentile = rolling_vol.rolling(window=window * 5).rank(pct=True)

        regime = np.where(
            vol_percentile > 0.8,
            "high",
            np.where(vol_percentile < 0.2, "low", "medium"),
        )

        return pd.Series(regime, index=ohlc_data.index)

    @staticmethod
    def _identify_gaps(ohlc_data, gap_threshold=0.01):
        """Identify price gaps."""
        gaps = []
        for i in range(1, len(ohlc_data)):
            prev_close = ohlc_data["close"].iloc[i - 1]
            current_open = ohlc_data["open"].iloc[i]

            gap_size = (current_open - prev_close) / prev_close

            if abs(gap_size) > gap_threshold:
                gaps.append(
                    {
                        "index": i,
                        "gap_size": gap_size,
                        "gap_type": "up" if gap_size > 0 else "down",
                        "prev_close": prev_close,
                        "current_open": current_open,
                    },
                )

        return gaps


class SentimentAnalyzer:
    """Analyze market sentiment from various sources."""

    def __init__(self, news_api_key=None):
        self.news_api_key = news_api_key
        self.sentiment_history = []

    def analyze_news_sentiment(self, symbol, days_back=7):
        """Analyze sentiment from financial news."""
        if not self.news_api_key:
            logger.info("No news API key provided, running in Sentiment Simulation Mode")
            return self._generate_mock_sentiment()

        try:
            # Fetch news articles
            articles = self._fetch_news_articles(symbol, days_back)

            # Analyze sentiment
            sentiments = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment_score = self._analyze_text_sentiment(text)
                sentiments.append(
                    {
                        "score": sentiment_score,
                        "title": article.get("title"),
                        "published_at": article.get("publishedAt"),
                        "source": article.get("source", {}).get("name"),
                    },
                )

            # Aggregate sentiment
            if sentiments:
                avg_sentiment = np.mean([s["score"] for s in sentiments])
                sentiment_trend = self._calculate_sentiment_trend(sentiments)

                return {
                    "average_sentiment": avg_sentiment,
                    "sentiment_trend": sentiment_trend,
                    "article_count": len(sentiments),
                    "positive_count": len([s for s in sentiments if s["score"] > 0.1]),
                    "negative_count": len([s for s in sentiments if s["score"] < -0.1]),
                    "neutral_count": len([s for s in sentiments if -0.1 <= s["score"] <= 0.1]),
                    "recent_articles": sentiments[:5],  # Top 5 recent articles
                }
            else:
                return self._generate_mock_sentiment()

        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return self._generate_mock_sentiment()

    def _fetch_news_articles(self, symbol, days_back):
        """Fetch news articles from API."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Example API call (replace with actual news API)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{symbol} stock market",
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "sortBy": "relevancy",
            "apiKey": self.news_api_key,
            "language": "en",
            "pageSize": 50,
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            return []

    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of text using a lightweight keyword heuristic."""
        try:
            normalized_text = (text or "").lower()
            positive_terms = (
                "gain",
                "bull",
                "bullish",
                "strong",
                "beat",
                "growth",
                "up",
                "optimistic",
                "rally",
                "positive",
                "surge",
            )
            negative_terms = (
                "loss",
                "bear",
                "bearish",
                "weak",
                "miss",
                "decline",
                "down",
                "pessimistic",
                "selloff",
                "negative",
                "drop",
            )

            positive_score = sum(term in normalized_text for term in positive_terms)
            negative_score = sum(term in normalized_text for term in negative_terms)
            total = positive_score + negative_score

            if total == 0:
                return 0.0

            return (positive_score - negative_score) / total
        except Exception:
            return 0.0

    def _calculate_sentiment_trend(self, sentiments):
        """Calculate sentiment trend over time."""
        if len(sentiments) < 2:
            return 0.0

        # Sort by publication date
        sorted_sentiments = sorted(sentiments, key=lambda x: x.get("published_at", ""))

        # Calculate trend using linear regression on sentiment scores
        scores = [s["score"] for s in sorted_sentiments]
        x = np.arange(len(scores))

        if len(scores) > 1:
            correlation = np.corrcoef(x, scores)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0

    def _generate_mock_sentiment(self) -> dict | None:
        """Return None — no live sentiment source configured."""
        logger.debug("No news API key configured; sentiment unavailable.")
        return None

    def analyze_social_sentiment(self, symbol: str) -> dict | None:
        """Return None — no social media API configured."""
        logger.debug("No social media API configured for %s; social sentiment unavailable.", symbol)
        return None


class FundamentalAnalyzer:
    """Analyze fundamental data and economic indicators."""

    def __init__(self, alpha_vantage_key=None):
        self.alpha_vantage_key = alpha_vantage_key

    def get_economic_indicators(self) -> dict | None:
        """Return None — no economic data API configured."""
        if not self.alpha_vantage_key:
            logger.debug("No Alpha Vantage API key; economic indicators unavailable.")
            return None
        # Real implementation would fetch from Alpha Vantage or FRED here
        return None

    def get_company_fundamentals(self, symbol: str) -> dict | None:
        """Return None — no fundamental data API configured."""
        if not self.alpha_vantage_key:
            logger.debug("No Alpha Vantage API key; fundamentals for %s unavailable.", symbol)
            return None
        # Real implementation would fetch from Alpha Vantage here
        return None


class MultiModalFeatureExtractor:
    """
    Comprehensive feature extractor combining technical, fundamental,
    sentiment, and market structure data.
    """

    def __init__(self, news_api_key=None, alpha_vantage_key=None):
        self.technical_indicators = TechnicalIndicators()
        self.market_analyzer = MarketStructureAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer(news_api_key)
        self.fundamental_analyzer = FundamentalAnalyzer(alpha_vantage_key)

        self.feature_cache = {}
        self.last_update = {}

    def extract_all_features(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        include_sentiment=True,
        include_fundamentals=True,
    ) -> dict:
        """
        Extract comprehensive features from all data sources.

        Args:
            symbol: Trading symbol
            ohlcv_data: OHLCV price data DataFrame
            include_sentiment: Whether to include sentiment analysis
            include_fundamentals: Whether to include fundamental data

        Returns:
            Dictionary with all extracted features
        """

        logger.info(f"Extracting features for {symbol}")

        features = {}

        # Technical features
        features["technical"] = self._extract_technical_features(ohlcv_data)

        # Market structure features
        features["market_structure"] = self._extract_market_structure_features(ohlcv_data)

        # Time-based features
        features["temporal"] = self._extract_temporal_features(ohlcv_data)

        # Volatility features
        features["volatility"] = self._extract_volatility_features(ohlcv_data)

        # Volume features
        features["volume"] = self._extract_volume_features(ohlcv_data)

        # Price action features
        features["price_action"] = self._extract_price_action_features(ohlcv_data)

        # Candlestick pattern features
        features["candlestick_patterns"] = self._extract_candlestick_patterns(ohlcv_data)

        # Sentiment features
        if include_sentiment:
            features["sentiment"] = self._extract_sentiment_features(symbol)

        # Fundamental features
        if include_fundamentals:
            features["fundamental"] = self._extract_fundamental_features(symbol)

        # Cross-asset features
        features["cross_asset"] = self._extract_cross_asset_features(symbol)

        # Feature engineering
        features["engineered"] = self._engineer_interaction_features(features)

        # Normalize features
        features["normalized"] = self._normalize_features(features)

        logger.info(f"Feature extraction completed for {symbol}")

        return features

    def _extract_technical_features(self, data: pd.DataFrame) -> dict:
        """Extract technical indicator features."""

        close = data["close"] if "close" in data.columns else data["Close"]
        high = data["high"] if "high" in data.columns else data["High"]
        low = data["low"] if "low" in data.columns else data["Low"]
        _volume = data["volume"] if "volume" in data.columns else data.get("Volume", pd.Series([1] * len(data)))

        features = {}

        # Trend indicators
        features["rsi_14"] = self.technical_indicators.calculate_rsi(close, 14).iloc[-1] if len(close) > 14 else 50
        features["rsi_21"] = self.technical_indicators.calculate_rsi(close, 21).iloc[-1] if len(close) > 21 else 50

        # MACD
        macd, signal, histogram = self.technical_indicators.calculate_macd(close)
        features["macd"] = macd.iloc[-1] if len(macd) > 0 else 0
        features["macd_signal"] = signal.iloc[-1] if len(signal) > 0 else 0
        features["macd_histogram"] = histogram.iloc[-1] if len(histogram) > 0 else 0

        # Bollinger Bands
        _bb_upper, _bb_lower, bb_width, bb_position = self.technical_indicators.calculate_bollinger_bands(close)
        features["bb_width"] = bb_width.iloc[-1] if len(bb_width) > 0 else 0
        features["bb_position"] = bb_position.iloc[-1] if len(bb_position) > 0 else 0.5

        # Stochastic
        stoch_k, stoch_d = self.technical_indicators.calculate_stochastic(high, low, close)
        features["stoch_k"] = stoch_k.iloc[-1] if len(stoch_k) > 0 else 50
        features["stoch_d"] = stoch_d.iloc[-1] if len(stoch_d) > 0 else 50

        # ATR
        features["atr"] = self.technical_indicators.calculate_atr(high, low, close).iloc[-1] if len(close) > 14 else 0

        # ADX
        adx, plus_di, minus_di = self.technical_indicators.calculate_adx(high, low, close)
        features["adx"] = adx.iloc[-1] if len(adx) > 0 else 25
        features["plus_di"] = plus_di.iloc[-1] if len(plus_di) > 0 else 25
        features["minus_di"] = minus_di.iloc[-1] if len(minus_di) > 0 else 25

        # Williams %R
        features["williams_r"] = (
            self.technical_indicators.calculate_williams_r(high, low, close).iloc[-1] if len(close) > 14 else -50
        )

        # CCI
        features["cci"] = self.technical_indicators.calculate_cci(high, low, close).iloc[-1] if len(close) > 20 else 0

        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(close) > period:
                ma = close.rolling(window=period).mean()
                features[f"sma_{period}"] = ma.iloc[-1]
                features[f"price_vs_sma_{period}"] = (close.iloc[-1] / ma.iloc[-1]) - 1

        # EMA
        for period in [12, 26]:
            if len(close) > period:
                ema = close.ewm(span=period).mean()
                features[f"ema_{period}"] = ema.iloc[-1]
                features[f"price_vs_ema_{period}"] = (close.iloc[-1] / ema.iloc[-1]) - 1

        return features

    def _extract_market_structure_features(self, data: pd.DataFrame) -> dict:
        """Extract market structure features."""

        patterns = self.market_analyzer.calculate_price_patterns(data)

        features = {}
        features["trend_strength"] = patterns["trend_strength"].iloc[-1] if len(patterns["trend_strength"]) > 0 else 0
        features["trend_direction"] = (
            patterns["trend_direction"].iloc[-1] if len(patterns["trend_direction"]) > 0 else 0
        )
        features["volatility_regime"] = (
            1
            if patterns["volatility_regime"].iloc[-1] == "high"
            else (0 if patterns["volatility_regime"].iloc[-1] == "medium" else -1)
            if len(patterns["volatility_regime"]) > 0
            else 0
        )
        features["support_resistance_ratio"] = len(patterns["support_levels"]) / max(
            len(patterns["resistance_levels"]),
            1,
        )
        features["gap_count"] = len(patterns["gaps"])

        return features

    def _extract_temporal_features(self, data: pd.DataFrame) -> dict:
        """Extract time-based features."""

        features = {}

        if "time" in data.columns or hasattr(data.index, "hour"):
            # Hour of day
            if hasattr(data.index, "hour"):
                features["hour"] = data.index[-1].hour
            else:
                features["hour"] = pd.to_datetime(data["time"].iloc[-1]).hour

            # Day of week
            if hasattr(data.index, "dayofweek"):
                features["day_of_week"] = data.index[-1].dayofweek
            else:
                features["day_of_week"] = pd.to_datetime(data["time"].iloc[-1]).dayofweek

            # Month
            if hasattr(data.index, "month"):
                features["month"] = data.index[-1].month
            else:
                features["month"] = pd.to_datetime(data["time"].iloc[-1]).month

            # Market session
            hour = features["hour"]
            if 9 <= hour < 12:
                features["market_session"] = 0  # Morning
            elif 12 <= hour < 15:
                features["market_session"] = 1  # Afternoon
            elif 15 <= hour < 16:
                features["market_session"] = 2  # Close
            else:
                features["market_session"] = 3  # After hours
        else:
            # Default values if time information is not available
            features["hour"] = 12
            features["day_of_week"] = 2
            features["month"] = 6
            features["market_session"] = 1

        return features

    def _extract_volatility_features(self, data: pd.DataFrame) -> dict:
        """Extract volatility-related features."""

        close = data["close"] if "close" in data.columns else data["Close"]
        high = data["high"] if "high" in data.columns else data["High"]
        low = data["low"] if "low" in data.columns else data["Low"]

        features = {}

        # Price returns
        returns = close.pct_change().dropna()

        if len(returns) > 1:
            # Realized volatility
            features["realized_volatility"] = returns.std()

            # Return skewness and kurtosis
            features["return_skewness"] = returns.skew()
            features["return_kurtosis"] = returns.kurtosis()

            # Volatility clustering
            features["vol_clustering"] = returns.rolling(window=10).std().std() if len(returns) > 10 else 0
        else:
            features["realized_volatility"] = 0
            features["return_skewness"] = 0
            features["return_kurtosis"] = 0
            features["vol_clustering"] = 0

        # Intraday volatility
        if len(high) > 0 and len(low) > 0:
            features["intraday_range"] = ((high - low) / close).iloc[-1]
            features["intraday_range_avg"] = (
                ((high - low) / close).rolling(window=20).mean().iloc[-1]
                if len(close) > 20
                else features["intraday_range"]
            )
        else:
            features["intraday_range"] = 0
            features["intraday_range_avg"] = 0

        return features

    def _extract_volume_features(self, data: pd.DataFrame) -> dict:
        """Extract volume-related features."""

        close = data["close"] if "close" in data.columns else data["Close"]
        high = data["high"] if "high" in data.columns else data["High"]
        low = data["low"] if "low" in data.columns else data["Low"]
        volume = data["volume"] if "volume" in data.columns else data.get("Volume", pd.Series([1] * len(data)))

        features = {}

        if len(volume) > 0 and volume.sum() > 0:
            # Volume indicators
            features["volume_ratio"] = (
                volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1] if len(volume) > 20 else 1
            )
            features["volume_trend"] = (
                volume.rolling(window=10).mean().iloc[-1] / volume.rolling(window=20).mean().iloc[-1]
                if len(volume) > 20
                else 1
            )

            # VWAP
            vwap = self.technical_indicators.calculate_vwap(high, low, close, volume)
            features["price_vs_vwap"] = (close.iloc[-1] / vwap.iloc[-1]) - 1 if len(vwap) > 0 else 0

            # Money Flow Index
            mfi = self.technical_indicators.calculate_mfi(high, low, close, volume)
            features["mfi"] = mfi.iloc[-1] if len(mfi) > 0 else 50

            # On Balance Volume
            obv = (np.sign(close.diff()) * volume).cumsum()
            features["obv_trend"] = (
                obv.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]).iloc[-1]
                if len(obv) > 10
                else 0
            )
        else:
            features["volume_ratio"] = 1
            features["volume_trend"] = 1
            features["price_vs_vwap"] = 0
            features["mfi"] = 50
            features["obv_trend"] = 0

        return features

    def _extract_price_action_features(self, data: pd.DataFrame) -> dict:
        """Extract price action features."""

        open_price = data["open"] if "open" in data.columns else data["Open"]
        close = data["close"] if "close" in data.columns else data["Close"]
        high = data["high"] if "high" in data.columns else data["High"]
        low = data["low"] if "low" in data.columns else data["Low"]

        features = {}

        if len(close) > 0:
            # Price changes
            features["price_change_1"] = close.pct_change(1).iloc[-1] if len(close) > 1 else 0
            features["price_change_5"] = close.pct_change(5).iloc[-1] if len(close) > 5 else 0
            features["price_change_20"] = close.pct_change(20).iloc[-1] if len(close) > 20 else 0

            # Candlestick patterns
            if len(open_price) > 0:
                body_size = abs(close - open_price) / close
                upper_shadow = (high - np.maximum(close, open_price)) / close
                lower_shadow = (np.minimum(close, open_price) - low) / close

                features["body_size"] = body_size.iloc[-1]
                features["upper_shadow"] = upper_shadow.iloc[-1]
                features["lower_shadow"] = lower_shadow.iloc[-1]
                features["body_position"] = (
                    ((close - low) / (high - low)).iloc[-1] if (high.iloc[-1] - low.iloc[-1]) > 0 else 0.5
                )
            else:
                features["body_size"] = 0
                features["upper_shadow"] = 0
                features["lower_shadow"] = 0
                features["body_position"] = 0.5

            # Price momentum
            features["momentum_10"] = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) > 10 else 0
            features["momentum_20"] = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0
        else:
            features["price_change_1"] = 0
            features["price_change_5"] = 0
            features["price_change_20"] = 0
            features["body_size"] = 0
            features["upper_shadow"] = 0
            features["lower_shadow"] = 0
            features["body_position"] = 0.5
            features["momentum_10"] = 0
            features["momentum_20"] = 0

        return features

    def _extract_sentiment_features(self, symbol: str) -> dict:
        """Extract sentiment features.

        Returns neutral defaults when sentiment APIs are not configured.
        """
        # News sentiment
        news_sentiment = self.sentiment_analyzer.analyze_news_sentiment(symbol)

        # Social sentiment
        social_sentiment = self.sentiment_analyzer.analyze_social_sentiment(symbol)

        features: dict = {}
        if news_sentiment is not None:
            features["news_sentiment"] = news_sentiment.get("average_sentiment", 0.0)
            features["news_trend"] = news_sentiment.get("sentiment_trend", 0.0)
            features["news_volume"] = news_sentiment.get("article_count", 0)
        else:
            features["news_sentiment"] = 0.0
            features["news_trend"] = 0.0
            features["news_volume"] = 0

        if social_sentiment is not None:
            features["twitter_sentiment"] = social_sentiment.get("twitter_sentiment", 0.0)
            features["reddit_sentiment"] = social_sentiment.get("reddit_sentiment", 0.0)
            features["social_volume"] = social_sentiment.get("social_volume", 0)
            features["mention_trend"] = social_sentiment.get("mention_trend", 0.0)
        else:
            features["twitter_sentiment"] = 0.0
            features["reddit_sentiment"] = 0.0
            features["social_volume"] = 0
            features["mention_trend"] = 0.0

        return features

    def _extract_fundamental_features(self, symbol: str) -> dict:
        """Extract fundamental features.

        Returns empty dict when fundamental data APIs are not configured.
        """
        features: dict = {}

        # Economic indicators
        economic_data = self.fundamental_analyzer.get_economic_indicators()
        if economic_data is not None:
            features.update(economic_data)

        # Company fundamentals
        company_data = self.fundamental_analyzer.get_company_fundamentals(symbol)
        if company_data is not None:
            features.update(company_data)

        return features

    def _extract_candlestick_patterns(self, data: pd.DataFrame) -> dict:
        """Extract candlestick pattern indicators."""
        try:
            patterns_df = get_all_patterns(data)
            if len(patterns_df) > 0:
                latest = patterns_df.iloc[-1]
                return {col: float(latest[col]) for col in patterns_df.columns}
        except Exception as e:
            logger.error(f"Error extracting candlestick patterns: {e}")

        return {
            "pattern_doji": 0,
            "pattern_hammer": 0,
            "pattern_inverted_hammer": 0,
            "pattern_bullish_engulfing": 0,
            "pattern_bearish_engulfing": 0,
            "pattern_marubozu": 0,
            "pattern_morning_star": 0,
            "pattern_evening_star": 0,
        }

    def _extract_cross_asset_features(self, symbol: str) -> dict:
        """Extract cross-asset correlation features.

        Returns empty dict — no cross-asset data source configured.
        """
        logger.debug("No cross-asset data source configured for %s.", symbol)
        return {}

    def _engineer_interaction_features(self, features: dict) -> dict:
        """Engineer interaction and derived features."""

        engineered = {}

        # Technical interactions
        if "technical" in features:
            tech = features["technical"]

            # RSI momentum
            engineered["rsi_momentum"] = tech.get("rsi_14", 50) - 50

            # MACD strength
            macd = tech.get("macd", 0)
            macd_signal = tech.get("macd_signal", 0)
            engineered["macd_strength"] = macd - macd_signal

            # Trend alignment
            if "market_structure" in features:
                _ms = features["market_structure"]
                engineered["trend_alignment"] = tech.get("plus_di", 25) - tech.get("minus_di", 25)

        # Sentiment-technical divergence
        if "sentiment" in features and "technical" in features:
            sentiment_score = features["sentiment"].get("news_sentiment", 0)
            price_momentum = features["technical"].get("momentum_10", 0)
            engineered["sentiment_momentum_divergence"] = sentiment_score - np.sign(price_momentum)

        # Volatility regime vs volume
        if "volatility" in features and "volume" in features:
            vol_regime = features["volatility"].get("realized_volatility", 0)
            volume_ratio = features["volume"].get("volume_ratio", 1)
            engineered["vol_volume_interaction"] = vol_regime * volume_ratio

        return engineered

    def _normalize_features(self, features: dict) -> dict:
        """Normalize features to standard ranges."""

        normalized: dict[str, Any] = {}

        # Define normalization ranges for different feature types
        # Normalization ranges for reference - normalization is done inline below
        # rsi: (0, 100), stoch: (0, 100), mfi: (0, 100), williams_r: (-100, 0)
        # percentage: (-1, 1), ratio: (0, 2), correlation: (-1, 1), sentiment: (-1, 1)

        for category, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                normalized[category] = {}
                for feature_name, value in feature_dict.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        # Determine normalization method based on feature name
                        if any(indicator in feature_name.lower() for indicator in ["rsi", "stoch", "mfi"]):
                            normalized[category][feature_name] = (value - 50) / 50  # Center around 0
                        elif "williams_r" in feature_name.lower():
                            normalized[category][feature_name] = (value + 50) / 50  # Center around 0
                        elif any(term in feature_name.lower() for term in ["sentiment", "correlation"]):
                            normalized[category][feature_name] = np.clip(value, -1, 1)
                        elif "ratio" in feature_name.lower():
                            normalized[category][feature_name] = np.clip((value - 1), -1, 1)
                        else:
                            # Standard normalization (assuming roughly normal distribution)
                            normalized[category][feature_name] = np.clip(value, -3, 3) / 3
                    else:
                        normalized[category][feature_name] = 0  # Default for invalid values

        return normalized

    def get_feature_vector(self, features: dict, feature_selection: Any = None) -> tuple[np.ndarray, list[str]]:
        """Convert features dictionary to a flattened feature vector."""

        if feature_selection is None:
            # Use all normalized features
            feature_selection = features.get("normalized", features)

        vector = []
        feature_names = []

        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, (int, float)) and not np.isnan(value):
                    vector.append(value)
                    feature_names.append(f"{prefix}{key}")
                else:
                    vector.append(0.0)  # Default for invalid values
                    feature_names.append(f"{prefix}{key}")

        flatten_dict(feature_selection)

        return np.array(vector), feature_names

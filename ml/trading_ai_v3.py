"""
DART v3.0 - State-of-the-Art AI Trading Engine
Fully refactored with LLM integration, proper evaluation pipeline,
and modern ML practices.
"""

import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler

# DART v3.0 SOTA updates
from config.dart_config import get_config
from ml.candlestick_patterns import get_all_patterns
from ml.deep_rl_agent import SoftActorCriticV2

logger = logging.getLogger("trading_ai_v3")


class FeatureEngineer:
    """SOTA feature engineering pipeline."""

    @staticmethod
    def engineer_features(df: pd.DataFrame, n_lags: int = 20) -> pd.DataFrame:
        """
        Create comprehensive features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            n_lags: Number of lag features to create

        Returns:
            DataFrame with engineered features
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Standardize columns
        df.columns = [col.lower() for col in df.columns]
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            logger.error(f"Missing required columns. Need: {required_cols}")
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # === Price-based features ===
        features["returns"] = df["close"].pct_change()
        features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        features["volatility"] = features["log_returns"].rolling(20).std()
        features["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
        features["close_open_ratio"] = (df["close"] - df["open"]) / df["open"]

        # === Momentum features ===
        for period in [5, 10, 20]:
            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

            # MACD with spans dynamically scaled to the period
            fast_span = max(2, int(period * 0.6))
            slow_span = max(3, int(period * 1.3))
            signal_span = max(2, int(period * 0.45))

            ema_fast = df["close"].ewm(span=fast_span).mean()
            ema_slow = df["close"].ewm(span=slow_span).mean()
            macd = ema_fast - ema_slow
            features[f"macd_{period}"] = macd
            features[f"macd_signal_{period}"] = macd.ewm(span=signal_span).mean()

        # === Trend features ===
        for period in [5, 10, 20]:
            features[f"sma_{period}"] = df["close"].rolling(period).mean()
            features[f"ema_{period}"] = df["close"].ewm(span=period).mean()
            features[f"trend_{period}"] = (df["close"] > features[f"sma_{period}"]).astype(int)

        # === Volatility features ===
        features["atr"] = FeatureEngineer._calculate_atr(df)
        features["bb_upper"], features["bb_lower"] = FeatureEngineer._calculate_bollinger(df)
        features["bb_squeeze"] = (features["bb_upper"] - features["bb_lower"]) / df["close"]

        # === Lag features ===
        for lag in range(1, min(n_lags + 1, len(df))):
            features[f"return_lag_{lag}"] = features["returns"].shift(lag)

        # === Volume features (if available) ===
        if "volume" in df.columns:
            features["volume_change"] = df["volume"].pct_change()
            features["volume_ma"] = df["volume"].rolling(20).mean()

        # === Candlestick Patterns ===
        patterns_df = get_all_patterns(df)
        for col in patterns_df.columns:
            features[col] = patterns_df[col]

        # Fill NaN values
        features = features.bfill().ffill()

        return features.dropna()

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def _calculate_bollinger(df: pd.DataFrame, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands."""
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, lower


class EnsembleModel:
    """SOTA ensemble model using stacking."""

    def __init__(self):
        """Initialize ensemble."""
        self.scaler = RobustScaler()
        self.model = None
        self._create_model()

    def _create_model(self):
        """Create stacking ensemble."""
        # Base models
        base_models = [
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)),
        ]

        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000)

        # Stacking classifier
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleModel:
        """Fit ensemble model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class TradingAIv3:
    """DART v3.0 Trading AI - State-of-the-Art Implementation."""

    def __init__(
        self,
        model_dir: str = "models",
        use_llm: bool = True,
        require_real_llm: bool = True,
        eval_pipeline=None,
    ):
        """
        Initialize TradingAI v3.0.

        Args:
            model_dir: Directory for model storage
            use_llm: Enable LLM market analysis
            eval_pipeline: Evaluation pipeline instance
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.use_llm = use_llm
        self.require_real_llm = require_real_llm
        self.eval_pipeline = eval_pipeline

        # Get configuration
        config = get_config()

        # Components
        self.feature_engineer = FeatureEngineer()
        self.ensemble_model: EnsembleModel | None = None
        self.llm_analyzer: Any = None

        # Try to load LLM analyzer
        if use_llm:
            try:
                from ml.llm_market_analyzer import LLMMarketAnalyzer

                self.llm_analyzer = LLMMarketAnalyzer(
                    model_name=config.ai.llm_model,
                    base_url=config.api.llm_base_url,
                    allow_fallback=not self.require_real_llm,
                )
                logger.info("LLM Market Analyzer loaded")
            except Exception as e:
                logger.warning(f"LLM not available: {e}")
                if self.require_real_llm:
                    raise
                self.use_llm = False

        # Initialize Deep RL Agent
        self.use_deep_rl = config.ai.use_deep_rl
        self.rl_agent: SoftActorCriticV2 | None = None
        if self.use_deep_rl:
            try:
                self.rl_agent = SoftActorCriticV2(
                    lr_actor=config.ai.rl_learning_rate,
                    lr_critic=config.ai.rl_learning_rate,
                    gamma=config.ai.rl_gamma,
                    tau=config.ai.rl_tau,
                )
                logger.info("Deep RL Agent (SAC v2.0) initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Deep RL Agent: {e}")
                self.use_deep_rl = False

        # Performance tracking
        self.training_history: list[dict] = []
        self.prediction_history: list[dict] = []
        self.last_strategy: dict[str, Any] | None = None
        self.model_trained_at: str | None = None

    def train(self, historical_data: list[dict]) -> dict:
        """
        Train model on historical data.

        Args:
            historical_data: List of OHLCV dictionaries

        Returns:
            Training metrics dictionary
        """
        logger.info("Starting model training...")
        start_time = time.time()

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)

        # Engineer features
        features_df = self.feature_engineer.engineer_features(df)

        if len(features_df) < 100:
            logger.error("Not enough data for training")
            return {}

        # Create target: 1 if next close > current close, 0 otherwise
        # Use integer positional indices to avoid misalignment after dropna()
        feature_positions = [df.index.get_loc(idx) for idx in features_df.index]
        close_at_features = df["close"].iloc[feature_positions].values
        close_next = df["close"].iloc[[min(p + 1, len(df) - 1) for p in feature_positions]].values
        y = (close_next[:-1] > close_at_features[:-1]).astype(int)
        X = features_df.iloc[:-1].values

        # Train ensemble
        self.ensemble_model = EnsembleModel()
        self.ensemble_model.fit(X, y)

        # Calculate training accuracy as a proxy/benchmark metric
        preds = self.ensemble_model.predict(X)
        accuracy = float(np.mean(preds == y)) if len(y) > 0 else 0.0

        # Train Deep RL Agent if active
        if self.use_deep_rl and self.rl_agent:
            try:
                logger.info("Training Deep RL Agent (SAC)...")
                # Ensure we have enough data to construct states and transitions
                if len(df) >= 100:
                    # Pre-calculate features on the full DataFrame to make RL state preparation fast
                    df_full = df.copy()
                    df_full.columns = [col.lower() for col in df_full.columns]
                    features_full = self.feature_engineer.engineer_features(df_full)

                    # Helper for fast state creation
                    def get_fast_state(t_idx):
                        start_idx = max(0, t_idx - 49)
                        slice_df = df_full.iloc[start_idx : t_idx + 1]
                        slice_features = features_full.iloc[start_idx : t_idx + 1]

                        feature_cols = [c for c in features_full.columns if c != "target"][:20]
                        technical = slice_features[feature_cols].values
                        if len(technical) < 50:
                            pad = np.zeros((50 - len(technical), 20))
                            technical = np.vstack([pad, technical])
                        else:
                            technical = technical[-50:]

                        price_cols = ["open", "high", "low", "close"]
                        price = slice_df[price_cols].values
                        if len(price) < 50:
                            pad = np.zeros((50 - len(price), 4))
                            price = np.vstack([pad, price])
                        else:
                            price = price[-50:]
                        price = (price - price.mean()) / (price.std() + 1e-10)

                        if "volume" in df_full.columns:
                            volume = slice_df["volume"].values
                            if len(volume) < 50:
                                pad = np.zeros(50 - len(volume))
                                volume = np.concatenate([pad, volume])
                            else:
                                volume = volume[-50:]
                            volume = (volume - volume.mean()) / (volume.std() + 1e-10)
                        else:
                            volume = np.zeros(50)

                        return {
                            "technical": technical.reshape(1, 50, -1)[:, :, :20],
                            "price": price.reshape(1, 50, 4),
                            "volume": volume.reshape(1, 50),
                        }

                    transitions_added = 0
                    for t in range(49, len(df_full) - 1):
                        state = get_fast_state(t)
                        next_state = get_fast_state(t + 1)

                        # Select action (with exploration noise)
                        action = self.rl_agent.select_action(state, eval_mode=False)

                        # Compute expected price change reward
                        curr_close = df_full["close"].iloc[t]
                        next_close = df_full["close"].iloc[t + 1]
                        price_change = (next_close - curr_close) / (curr_close + 1e-10)
                        reward = float(action[0] * price_change * 100.0)

                        done = (t == len(df_full) - 2)

                        # Store in experience replay
                        self.rl_agent.store_transition(state, action, reward, next_state, done, 0)
                        transitions_added += 1

                    # Run training updates
                    batch_size = 256
                    if len(self.rl_agent.replay_buffer) >= batch_size:
                        train_steps = min(50, transitions_added // 2)
                        train_steps = max(10, train_steps)
                        logger.info("Running %d SAC training steps (buffer size: %d)...", train_steps, len(self.rl_agent.replay_buffer))
                        for _ in range(train_steps):
                            self.rl_agent.train(batch_size=batch_size)
                    else:
                        logger.warning(f"Replay buffer size ({len(self.rl_agent.replay_buffer)}) < batch size ({batch_size}). Skipping training steps.")
            except Exception as e:
                logger.error(f"Error training Deep RL Agent: {e}", exc_info=True)

        # Store training info
        self.model_trained_at = datetime.now().isoformat()
        training_time = time.time() - start_time

        metrics = {
            "status": "success",
            "samples": len(X),
            "training_time_s": training_time,
            "features": len(X[0]),
            "trained_at": self.model_trained_at,
            "accuracy": accuracy,
            "win_rate": accuracy,
        }

        logger.info("Training completed in %.2fs | Accuracy: %.2f%%", training_time, accuracy * 100)
        self.training_history.append(metrics)

        return metrics

    def _prepare_rl_state(self, current_data: list[dict]) -> dict:
        """Prepare state for Deep RL agent."""
        df = pd.DataFrame(current_data)
        df.columns = [col.lower() for col in df.columns]

        # Prepare technical features
        features_df = self.feature_engineer.engineer_features(df)
        if len(features_df) < 50:
            technical = np.zeros((50, 20))
        else:
            feature_cols = [c for c in features_df.columns if c != "target"][:20]
            technical = features_df[feature_cols].tail(50).values
            if len(technical) < 50:
                pad = np.zeros((50 - len(technical), technical.shape[1]))
                technical = np.vstack([pad, technical])

        # Prepare price data (OHLC)
        price_cols = ["open", "high", "low", "close"]
        available_cols = [c for c in price_cols if c in df.columns]

        if len(available_cols) == 4:
            price = df[available_cols].tail(50).values
            if len(price) < 50:
                pad = np.zeros((50 - len(price), 4))
                price = np.vstack([pad, price])
            price = (price - price.mean()) / (price.std() + 1e-10)
        else:
            price = np.zeros((50, 4))

        # Prepare volume data
        if "volume" in df.columns:
            volume = df["volume"].tail(50).values
            if len(volume) < 50:
                pad = np.zeros(50 - len(volume))
                volume = np.concatenate([pad, volume])
            volume = (volume - volume.mean()) / (volume.std() + 1e-10)
        else:
            volume = np.zeros(50)

        return {
            "technical": technical.reshape(1, 50, -1)[:, :, :20],
            "price": price.reshape(1, 50, 4),
            "volume": volume.reshape(1, 50),
        }

    def generate_strategy(
        self,
        current_data: list[dict],
        use_llm: bool = True,
        granularity: int = 60,
    ) -> dict | None:
        """
        Generate trading strategy.

        Args:
            current_data: Recent OHLCV data
            use_llm: Include LLM analysis
            granularity: Candle timeframe in seconds (60, 300, 900, 3600, etc.)

        Returns:
            Strategy dictionary or None
        """
        if self.ensemble_model is None:
            logger.warning("Model not trained")
            return None

        logger.info("Generating trading strategy...")

        # Convert to DataFrame
        df = pd.DataFrame(current_data)

        # Engineer features
        features_df = self.feature_engineer.engineer_features(df)

        if len(features_df) < 1:
            logger.warning("Insufficient data for feature engineering")
            return None

        # Get latest features
        X_latest = features_df.iloc[-1:].values

        # Model prediction
        prediction = int(self.ensemble_model.predict(X_latest)[0])
        proba = self.ensemble_model.predict_proba(X_latest)[0]

        confidence = float(proba[prediction])
        volatility = float(features_df["volatility"].iloc[-1]) if "volatility" in features_df.columns else 0.02

        # Initialize RL variables
        rl_prediction = None
        rl_confidence = 0.0
        regime_info = None
        uncertainty_info = None
        rl_action = None

        if self.use_deep_rl and self.rl_agent:
            try:
                # Prepare state for RL
                rl_state = self._prepare_rl_state(current_data)
                rl_action = self.rl_agent.select_action(rl_state, eval_mode=True)
                regime_info = self.rl_agent.detect_market_regime(rl_state)
                uncertainty_info = self.rl_agent.get_uncertainty(rl_state)

                if isinstance(uncertainty_info, dict):
                    uncertainty_info["total"] = uncertainty_info.get("uncertainty", 0.15)

                rl_prediction = 1 if rl_action[0] > 0 else 0
                rl_confidence = max(0.0, min(1.0, 1.0 - uncertainty_info["uncertainty"]))
                logger.info("RL agent prediction: %s with confidence %.2f", rl_prediction, rl_confidence)
            except Exception as e:
                logger.warning(f"RL prediction failed: {e}")

        # Combine predictions with weighted average
        if rl_prediction is not None:
            # Stacking classifier confidence + RL confidence
            ml_weight = confidence
            rl_weight = rl_confidence * 0.5

            total_weight = ml_weight + rl_weight
            if total_weight > 0:
                combined_prob = (prediction * ml_weight + rl_prediction * rl_weight) / total_weight
                final_prediction = 1 if combined_prob > 0.5 else 0
                final_confidence = (confidence * ml_weight + rl_confidence * rl_weight) / total_weight
            else:
                final_prediction = prediction
                final_confidence = confidence
        else:
            final_prediction = prediction
            final_confidence = confidence

        final_direction = "CALL" if final_prediction == 1 else "PUT"

        # ---------------------------------------------------------------
        # Smart Duration Selection
        # Uses granularity (analysis timeframe), volatility, confidence,
        # market regime, and the RL policy's action dimension to pick the 
        # optimal trade duration dynamically.
        # ---------------------------------------------------------------

        # Base duration mapping: analysis timeframe -> recommended trade durations
        granularity_duration_map = {
            60:    {"min": 5,    "base": 120,   "max": 1800},   # 1m candles  -> 5s to 30m trades
            300:   {"min": 15,   "base": 300,   "max": 3600},   # 5m candles  -> 15s to 1h trades
            900:   {"min": 30,   "base": 900,   "max": 7200},   # 15m candles -> 30s to 2h trades
            3600:  {"min": 60,   "base": 3600,  "max": 28800},  # 1h candles  -> 1m to 8h trades
            14400: {"min": 300,  "base": 14400, "max": 86400},  # 4h candles  -> 5m to 24h trades
            86400: {"min": 1800, "base": 86400, "max": 432000}, # 1d candles  -> 30m to 5d trades
        }

        # Find the closest matching granularity
        available_granularities = sorted(granularity_duration_map.keys())
        closest_gran = min(available_granularities, key=lambda g: abs(g - granularity))
        dur_range = granularity_duration_map[closest_gran]

        base_duration = dur_range["base"]
        min_duration = dur_range["min"]
        max_duration = dur_range["max"]

        # 1. Volatility adjustment (continuous scale, anchored at 0.005 base volatility)
        vol_factor = 0.005 / (volatility + 1e-8)
        vol_factor = max(0.2, min(3.0, vol_factor))

        # 2. Confidence adjustment (continuous scale, higher confidence = faster entry/exit)
        conf_factor = 1.5 - final_confidence
        conf_factor = max(0.5, min(2.0, conf_factor))

        # 3. Regime adjustment
        regime_factor = 1.0
        if regime_info:
            regime_name = regime_info.get("regime_name", "Normal").lower()
            if "trending" in regime_name or "bull" in regime_name or "bear" in regime_name:
                regime_factor = 1.5   # Trending markets -> longer to capture the move
            elif "volatile" in regime_name or "crisis" in regime_name:
                regime_factor = 0.5   # Volatile/crisis -> shorter trades
            elif "ranging" in regime_name or "sideways" in regime_name:
                regime_factor = 0.75  # Ranging -> shorter, less directional

        # 4. Uncertainty adjustment (continuous scale)
        total_uncertainty = 1.0 - rl_confidence if rl_prediction is not None else 0.3
        uncertainty_factor = 0.5 + total_uncertainty
        uncertainty_factor = max(0.5, min(2.0, uncertainty_factor))

        # 5. RL Action Duration Adjustment
        rl_factor = 1.0
        if rl_prediction is not None and rl_action is not None and len(rl_action) > 1:
            # Map action[1] from [-1, 1] to [0.36, 2.71] using exponential
            rl_factor = float(math.exp(rl_action[1]))

        # Calculate final duration
        duration = int(base_duration * vol_factor * conf_factor * regime_factor * uncertainty_factor * rl_factor)

        # Clamp to the valid range for this timeframe
        duration = max(min_duration, min(max_duration, duration))

        # Round to clean values for better readability
        if duration >= 3600:
            duration = round(duration / 300) * 300   # Round to nearest 5 min
        elif duration >= 300:
            duration = round(duration / 60) * 60     # Round to nearest minute
        elif duration >= 60:
            duration = round(duration / 5) * 5       # Round to nearest 5s
        else:
            duration = max(5, duration)              # Ensure minimum of 5 seconds

        # Choose duration unit (use minutes for durations >= 120s for cleaner display)
        if duration >= 120 and duration % 60 == 0:
            duration_value = duration // 60
            duration_unit = "m"
        else:
            duration_value = duration
            duration_unit = "s"

        logger.info(
            f"AI Duration: {duration_value}{duration_unit} "
            f"(granularity={granularity}s, vol={volatility:.4f}*F{vol_factor:.2f}, "
            f"conf={final_confidence:.2f}*F{conf_factor:.2f}, "
            f"regime_f={regime_factor:.2f}, unc_f={uncertainty_factor:.2f}, rl_f={rl_factor:.2f})"
        )

        strategy = {
            "direction": final_direction,
            "confidence": float(final_confidence),
            "probability": float(proba[1]),
            "duration": duration_value,
            "duration_unit": duration_unit,
            "contract_type": final_direction,
            "volatility": float(volatility),
            "timestamp": datetime.now().isoformat(),
            "model": "ensemble_v3",
            "uncertainty": uncertainty_info
            if uncertainty_info
            else {"epistemic": 0.05, "aleatoric": 0.05, "total": 0.10},
        }

        if regime_info:
            strategy["market_regime"] = regime_info["regime_name"]
            strategy["regime_confidence"] = float(regime_info["confidence"])

        # LLM Enhancement
        if use_llm and self.llm_analyzer:
            try:
                llm_analysis = self.llm_analyzer.analyze_market_data(
                    price_data=df,
                    technical_indicators=self._extract_technical_indicators(features_df.iloc[-1]),
                )
                strategy["llm_analysis"] = llm_analysis
                # Adjust confidence using LLM (normalize 0-100 score to 0-1)
                llm_conf = llm_analysis.get("confidence", 50) / 100.0
                strategy["confidence"] = (final_confidence + llm_conf) / 2
            except Exception as e:
                logger.debug(f"LLM analysis failed: {e}")

        self.last_strategy = strategy
        self.prediction_history.append(strategy)

        return strategy

    def _extract_technical_indicators(self, features: pd.Series) -> dict:
        """Extract technical indicators from features."""
        return {key: float(value) for key, value in features.items() if isinstance(value, (int, float, np.number))}

    def update_from_trade_results(self, trade_results: list[dict]) -> None:
        """Update model from trade results."""
        if not trade_results or not self.eval_pipeline:
            return

        # Evaluate trading performance
        metrics = self.eval_pipeline.evaluate_trading_performance(trade_results)
        logger.info("Trade metrics: WR=%.1f%%, PF=%.2f", metrics.win_rate * 100, metrics.profit_factor)

    def save_model(self, name: str = "model") -> Path | None:
        """Save model to disk."""
        if self.ensemble_model is None:
            logger.warning("No model to save")
            return None

        filepath = self.model_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(self.ensemble_model, filepath)
        logger.info("Model saved to %s", filepath)
        return filepath

    def load_model(self, filepath: Path) -> bool:
        """Load model from disk."""
        try:
            self.ensemble_model = joblib.load(filepath)
            logger.info("Model loaded from %s", filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    @property
    def strategy(self) -> dict | None:
        """Return the last generated strategy for dashboard compatibility."""
        return self.last_strategy

    @property
    def model(self):
        """Return the trained ensemble model for dashboard compatibility."""
        return self.ensemble_model

    @property
    def performance_metrics(self) -> dict[str, float]:
        """Return performance metrics dictionary for dashboard compatibility."""
        accuracy = 0.0
        win_rate = 0.0
        if self.training_history:
            latest = self.training_history[-1]
            accuracy = latest.get("accuracy", accuracy)
            win_rate = latest.get("win_rate", win_rate)
        return {
            "accuracy": accuracy,
            "win_rate": win_rate,
        }

    @property
    def model_accuracy(self) -> float:
        """Return model accuracy for dashboard compatibility."""
        return self.performance_metrics["accuracy"]

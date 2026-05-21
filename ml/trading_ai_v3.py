"""
DART v3.0 - State-of-the-Art AI Trading Engine
Fully refactored with LLM integration, proper evaluation pipeline,
and modern ML practices.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

# DART v3.0 SOTA updates
from config.dart_config import get_config
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

            # MACD
            ema_fast = df["close"].ewm(span=12).mean()
            ema_slow = df["close"].ewm(span=26).mean()
            macd = ema_fast - ema_slow
            features[f"macd_{period}"] = macd
            features[f"macd_signal_{period}"] = macd.ewm(span=9).mean()

        # === Trend features ===
        for period in [5, 10, 20]:
            features[f"sma_{period}"] = df["close"].rolling(period).mean()
            features[f"ema_{period}"] = df["close"].ewm(span=period).mean()
            features[f"trend_{period}"] = (
                (df["close"] > features[f"sma_{period}"]).astype(int)
            )

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

        # Fill NaN values
        features = features.fillna(method="bfill").fillna(method="ffill")

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
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
        self.ensemble_model = None
        self.llm_analyzer = None

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
        self.rl_agent = None
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
        self.training_history = []
        self.prediction_history = []
        self.last_strategy = None
        self.model_trained_at = None

    def train(self, historical_data: List[Dict]) -> Dict:
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
        y = (df["close"].iloc[features_df.index.values].shift(-1) > df["close"].iloc[features_df.index.values]).astype(int).values[:-1]
        X = features_df.iloc[:-1].values

        # Train ensemble
        self.ensemble_model = EnsembleModel()
        self.ensemble_model.fit(X, y)

        # Store training info
        self.model_trained_at = datetime.now().isoformat()
        training_time = time.time() - start_time

        metrics = {
            "status": "success",
            "samples": len(X),
            "training_time_s": training_time,
            "features": len(X[0]),
            "trained_at": self.model_trained_at,
        }

        logger.info(f"Training completed in {training_time:.2f}s")
        self.training_history.append(metrics)

        return metrics

    def _prepare_rl_state(self, current_data: List[Dict]) -> Dict:
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
        current_data: List[Dict],
        use_llm: bool = True,
    ) -> Optional[Dict]:
        """
        Generate trading strategy.

        Args:
            current_data: Recent OHLCV data
            use_llm: Include LLM analysis

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
        prediction = self.ensemble_model.predict(X_latest)[0]
        proba = self.ensemble_model.predict_proba(X_latest)[0]

        confidence = float(proba[prediction])
        volatility = float(features_df["volatility"].iloc[-1]) if "volatility" in features_df.columns else 0.02

        # Initialize RL variables
        rl_prediction = None
        rl_confidence = 0.0
        regime_info = None

        if self.use_deep_rl and self.rl_agent:
            try:
                # Prepare state for RL
                rl_state = self._prepare_rl_state(current_data)
                rl_action = self.rl_agent.select_action(rl_state, eval_mode=True)
                regime_info = self.rl_agent.detect_market_regime(rl_state)
                uncertainty_info = self.rl_agent.get_uncertainty(rl_state)

                rl_prediction = 1 if rl_action[0] > 0 else 0
                rl_confidence = max(0.0, min(1.0, 1.0 - uncertainty_info["uncertainty"]))
                logger.info(f"RL agent prediction: {rl_prediction} with confidence {rl_confidence:.2f}")
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

        # Duration based on volatility and confidence
        if volatility > 0.015:
            base_duration = 30
        elif volatility > 0.01:
            base_duration = 60
        elif volatility > 0.005:
            base_duration = 120
        else:
            base_duration = 180

        # Adjust duration based on uncertainty
        total_uncertainty = 1.0 - rl_confidence if rl_prediction is not None else 0.3
        if total_uncertainty > 0.5:
            duration = int(base_duration * 1.3)
        elif total_uncertainty < 0.2:
            duration = int(base_duration * 0.8)
        else:
            duration = base_duration
        duration = max(30, min(300, duration))

        strategy = {
            "direction": final_direction,
            "confidence": float(final_confidence),
            "probability": float(proba[1]),
            "duration": duration,
            "duration_unit": "s",
            "contract_type": final_direction,
            "volatility": float(volatility),
            "timestamp": datetime.now().isoformat(),
            "model": "ensemble_v3",
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
                # Adjust confidence using LLM
                strategy["confidence"] = (final_confidence + llm_analysis.get("confidence", 0.5) * 0.01) / 2
            except Exception as e:
                logger.debug(f"LLM analysis failed: {e}")

        self.last_strategy = strategy
        self.prediction_history.append(strategy)

        return strategy

    def _extract_technical_indicators(self, features: pd.Series) -> Dict:
        """Extract technical indicators from features."""
        return {
            key: float(value)
            for key, value in features.items()
            if isinstance(value, (int, float, np.number))
        }

    def update_from_trade_results(self, trade_results: List[Dict]) -> None:
        """Update model from trade results."""
        if not trade_results or not self.eval_pipeline:
            return

        # Evaluate trading performance
        metrics = self.eval_pipeline.evaluate_trading_performance(trade_results)
        logger.info(f"Trade metrics: WR={metrics.win_rate:.1%}, PF={metrics.profit_factor:.2f}")

    def save_model(self, name: str = "model") -> Path:
        """Save model to disk."""
        if self.ensemble_model is None:
            logger.warning("No model to save")
            return None

        filepath = self.model_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(self.ensemble_model, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath: Path) -> bool:
        """Load model from disk."""
        try:
            self.ensemble_model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

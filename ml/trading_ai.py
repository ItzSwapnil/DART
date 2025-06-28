"""
DART v2.0 - Advanced AI Trading System
Enhanced with ensemble learning, model stacking, uncertainty quantification,
and deep RL integration.
"""

import datetime
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("trading_ai")

# Import enhanced components
ENHANCED_COMPONENTS_AVAILABLE = False
try:
    from ml.deep_rl_agent import SoftActorCriticV2 as SoftActorCritic
    from ml.feature_extractor import MultiModalFeatureExtractor
    from ml.risk_manager import AdvancedRiskManager  # RiskLevel, MarketCondition removed - unused

    ENHANCED_COMPONENTS_AVAILABLE = True
    logger.info("DART v2.0: Enhanced AI components loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced components not available: {e}")


class UncertaintyEstimator:
    """Estimate prediction uncertainty using ensemble disagreement and Monte Carlo dropout."""

    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        """Train multiple models with different random seeds."""
        self.models = []
        for i in range(self.n_estimators):
            model = GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=5, random_state=i * 42,
            )
            # Bootstrap sampling
            indices = np.random.choice(len(X), size=len(X), replace=True)
            model.fit(
                X.iloc[indices] if hasattr(X, "iloc") else X[indices],
                y.iloc[indices] if hasattr(y, "iloc") else y[indices],
            )
            self.models.append(model)
        return self

    def predict_with_uncertainty(self, X):
        """Predict with uncertainty estimation."""
        predictions = np.array([model.predict_proba(X) for model in self.models])

        # Mean prediction
        mean_pred = predictions.mean(axis=0)

        # Epistemic uncertainty (model disagreement)
        epistemic = predictions.std(axis=0)

        # Aleatoric uncertainty (inherent data noise) - approximated by entropy
        aleatoric = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)

        # Total uncertainty
        total_uncertainty = epistemic.mean(axis=1) + aleatoric

        return {
            "predictions": mean_pred.argmax(axis=1),
            "probabilities": mean_pred,
            "epistemic_uncertainty": epistemic.mean(axis=1),
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": total_uncertainty,
            "confidence": 1 - np.clip(total_uncertainty, 0, 1),
        }


class TradingAI:
    """DART v2.0 AI system for market analysis and trading strategy generation."""

    def __init__(self, model_dir="models", use_deep_rl=True, use_enhanced_features=True):
        """Initialize the TradingAI v2.0 system."""
        self.model_dir = model_dir
        self.use_deep_rl = use_deep_rl and ENHANCED_COMPONENTS_AVAILABLE
        self.use_enhanced_features = use_enhanced_features and ENHANCED_COMPONENTS_AVAILABLE

        # v2.0 Ensemble components
        self.ensemble_model = None
        self.stacking_model = None
        self.uncertainty_estimator = None
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.last_training_time = None
        self.strategy = None

        # Performance tracking
        self.performance_metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "win_rate": 0,
            "trades_count": 0,
            "model_confidence": 0,
            "uncertainty_mean": 0,
        }

        # Enhanced components (if available)
        if self.use_deep_rl:
            logger.info("Initializing SAC v2.0 Deep RL Agent...")
            self.rl_agent = SoftActorCritic(use_curiosity=True, n_steps=3)
            self.rl_mode = False  # Start with ensemble ML
        else:
            self.rl_agent = None

        if self.use_enhanced_features:
            logger.info("Initializing Enhanced Feature Extractor...")
            self.feature_extractor = MultiModalFeatureExtractor()
        else:
            self.feature_extractor = None

        if ENHANCED_COMPONENTS_AVAILABLE:
            logger.info("Initializing Advanced Risk Manager...")
            self.risk_manager = AdvancedRiskManager()
        else:
            self.risk_manager = None

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Feature importance and regime tracking
        self.feature_importance_history = []
        self.market_regime_history = []
        self.prediction_history = []

        logger.info("DART v2.0 TradingAI initialized successfully")

    def _prepare_features(self, df):
        """Prepare comprehensive features for model training from raw candle data."""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        # Standardize column names
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "epoch": "epoch",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "time" in df.columns:
            df = df.set_index("time")

        if len(df) < 30:
            logger.warning(f"Not enough data for feature calculation: {len(df)} rows")
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # === Price-based features ===
        features["price_change"] = df["Close"].pct_change()
        features["price_change_2"] = df["Close"].pct_change(2)
        features["price_change_5"] = df["Close"].pct_change(5)
        features["price_change_10"] = df["Close"].pct_change(10)

        # Log returns (more stable)
        features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        features["log_return_5"] = np.log(df["Close"] / df["Close"].shift(5))

        # Volatility features
        features["high_low_diff"] = (df["High"] - df["Low"]) / df["Open"]
        features["close_open_diff"] = (df["Close"] - df["Open"]) / df["Open"]
        features["rolling_volatility_5"] = features["log_return"].rolling(5).std()
        features["rolling_volatility_20"] = features["log_return"].rolling(20).std()
        features["volatility_ratio"] = features["rolling_volatility_5"] / (
            features["rolling_volatility_20"] + 1e-10
        )

        # === Trend Indicators ===
        macd = MACD(close=df["Close"])
        features["macd"] = macd.macd()
        features["macd_signal"] = macd.macd_signal()
        features["macd_diff"] = macd.macd_diff()
        features["macd_histogram"] = features["macd_diff"]

        # Moving averages
        for window in [5, 10, 20, 50]:
            sma = SMAIndicator(close=df["Close"], window=window)
            ema = EMAIndicator(close=df["Close"], window=window)
            features[f"sma_{window}"] = sma.sma_indicator()
            features[f"ema_{window}"] = ema.ema_indicator()
            features[f"price_sma_{window}_ratio"] = df["Close"] / (
                features[f"sma_{window}"] + 1e-10
            )

        # MA crossovers
        features["sma_5_10_cross"] = (features["sma_5"] > features["sma_10"]).astype(int)
        features["sma_10_20_cross"] = (features["sma_10"] > features["sma_20"]).astype(int)
        features["ema_5_10_cross"] = (features["ema_5"] > features["ema_10"]).astype(int)

        # ADX for trend strength
        adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"])
        features["adx"] = adx.adx()
        features["adx_pos"] = adx.adx_pos()
        features["adx_neg"] = adx.adx_neg()

        # === Momentum Indicators ===
        rsi = RSIIndicator(close=df["Close"])
        features["rsi"] = rsi.rsi()
        features["rsi_oversold"] = (features["rsi"] < 30).astype(int)
        features["rsi_overbought"] = (features["rsi"] > 70).astype(int)

        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"])
        features["stoch_k"] = stoch.stoch()
        features["stoch_d"] = stoch.stoch_signal()
        features["stoch_cross"] = (features["stoch_k"] > features["stoch_d"]).astype(int)

        williams = WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"])
        features["williams_r"] = williams.williams_r()

        # === Volatility Indicators ===
        bb = BollingerBands(close=df["Close"])
        features["bb_high"] = bb.bollinger_hband()
        features["bb_low"] = bb.bollinger_lband()
        features["bb_mid"] = bb.bollinger_mavg()
        features["bb_width"] = (features["bb_high"] - features["bb_low"]) / (
            features["bb_mid"] + 1e-10
        )
        features["bb_pct"] = (df["Close"] - features["bb_low"]) / (
            features["bb_high"] - features["bb_low"] + 1e-10
        )

        atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"])
        features["atr"] = atr.average_true_range()
        features["atr_ratio"] = features["atr"] / df["Close"]

        keltner = KeltnerChannel(high=df["High"], low=df["Low"], close=df["Close"])
        features["keltner_high"] = keltner.keltner_channel_hband()
        features["keltner_low"] = keltner.keltner_channel_lband()
        features["keltner_squeeze"] = (
            (features["bb_high"] < features["keltner_high"])
            & (features["bb_low"] > features["keltner_low"])
        ).astype(int)

        # === Volume Indicators ===
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
            features["obv"] = obv.on_balance_volume()
            features["obv_change"] = features["obv"].pct_change()

            vwap = VolumeWeightedAveragePrice(
                high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"],
            )
            features["vwap"] = vwap.volume_weighted_average_price()
            features["price_vwap_ratio"] = df["Close"] / (features["vwap"] + 1e-10)

            mfi = MFIIndicator(
                high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"],
            )
            features["mfi"] = mfi.money_flow_index()

            features["volume_change"] = df["Volume"].pct_change()
            features["volume_ma_5"] = df["Volume"].rolling(window=5).mean()
            features["volume_ma_20"] = df["Volume"].rolling(window=20).mean()
            features["volume_ratio"] = features["volume_ma_5"] / (features["volume_ma_20"] + 1e-10)

        # === Market Structure Features ===
        # Support/Resistance proximity
        features["high_20"] = df["High"].rolling(20).max()
        features["low_20"] = df["Low"].rolling(20).min()
        features["high_proximity"] = (features["high_20"] - df["Close"]) / (
            features["high_20"] - features["low_20"] + 1e-10
        )
        features["low_proximity"] = (df["Close"] - features["low_20"]) / (
            features["high_20"] - features["low_20"] + 1e-10
        )

        # Price momentum
        features["momentum_5"] = df["Close"] - df["Close"].shift(5)
        features["momentum_10"] = df["Close"] - df["Close"].shift(10)
        features["momentum_ratio"] = features["momentum_5"] / (features["momentum_10"] + 1e-10)

        # === Target variable ===
        features["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        # Drop NaN values
        features = features.dropna()

        return features

    def _create_stacking_ensemble(self):
        """Create a stacking ensemble with meta-learner."""
        # Base learners with diverse architectures
        base_learners = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42,
                ),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
            ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
        ]

        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        # Create stacking classifier
        stacking = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            stack_method="predict_proba",
            passthrough=True,  # Include original features
        )

        return stacking

    def train_model(self, historical_data):
        """Train v2.0 ensemble model with stacking and uncertainty estimation."""
        logger.info("DART v2.0: Starting enhanced model training...")

        # Prepare features
        features_df = self._prepare_features(historical_data)

        if len(features_df) < 100:
            logger.warning(f"Insufficient data for training: {len(features_df)} samples")
            return None

        # Split features and target
        X = features_df.drop("target", axis=1)
        y = features_df["target"]

        # Use time-series aware splitting
        tscv = TimeSeriesSplit(n_splits=5)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

        # === Train Stacking Ensemble ===
        logger.info("Training stacking ensemble with meta-learner...")
        self.stacking_model = self._create_stacking_ensemble()
        self.stacking_model.fit(X_train_df, y_train)

        # Calibrate probabilities (using cross-validation for scikit-learn 1.5+)
        logger.info("Calibrating prediction probabilities...")
        # Note: cv="prefit" was deprecated in sklearn 1.5, using cv=5 instead
        self.model = CalibratedClassifierCV(
            self.stacking_model, cv=5, method="isotonic",
        )
        self.model.fit(X_train_df, y_train)

        # === Train Uncertainty Estimator ===
        logger.info("Training uncertainty estimator...")
        self.uncertainty_estimator = UncertaintyEstimator(n_estimators=10)
        self.uncertainty_estimator.fit(X_train_df, y_train)

        # === Evaluate ===
        y_pred = self.model.predict(X_val_df)
        y_proba = self.model.predict_proba(X_val_df)

        # Uncertainty analysis
        uncertainty_results = self.uncertainty_estimator.predict_with_uncertainty(X_val_df)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        # Cross-validation with time series split
        cv_scores = cross_val_score(
            self.stacking_model, X_train_df, y_train, cv=tscv, scoring="precision",
        )
        cv_precision = cv_scores.mean()

        # Feature importance from stacking model
        try:
            rf_model = self.stacking_model.named_estimators_["rf"]
            feature_importance = pd.Series(
                rf_model.feature_importances_, index=X.columns,
            ).sort_values(ascending=False)
            self.feature_importance_history.append(feature_importance.head(20).to_dict())
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        # Store metrics
        self.performance_metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "cv_precision": float(cv_precision),
            "win_rate": float(precision),
            "trades_count": 0,
            "model_confidence": float(y_proba.max(axis=1).mean()),
            "uncertainty_mean": float(uncertainty_results["total_uncertainty"].mean()),
            "model_type": "stacking_ensemble_v2",
            "n_features": X.shape[1],
            "training_samples": len(X_train),
        }

        self.last_training_time = datetime.datetime.now()

        # Save model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"dart_v2_ensemble_{timestamp}.joblib")
        joblib.dump(
            {
                "model": self.model,
                "stacking_model": self.stacking_model,
                "scaler": self.scaler,
                "metrics": self.performance_metrics,
            },
            model_path,
        )

        logger.info(
            f"DART v2.0 Training complete - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Uncertainty: {self.performance_metrics['uncertainty_mean']:.4f}",
        )

        return self.performance_metrics

    def generate_strategy(self, current_data, recalculate=False):
        """Generate trading strategy with uncertainty-aware confidence."""
        if self.model is None:
            logger.warning("No trained model available")
            return None

        if self.strategy is not None and not recalculate:
            return self.strategy

        logger.info("DART v2.0: Generating trading strategy...")

        # Prepare features
        features_df = self._prepare_features(current_data)
        if len(features_df) < 1:
            logger.warning(f"No data after feature prep: {len(features_df)}")
            return None

        # Get latest features
        latest_features = features_df.iloc[-1:].drop("target", axis=1)
        latest_scaled = self.scaler.transform(latest_features)
        latest_df = pd.DataFrame(latest_scaled, columns=latest_features.columns)

        # Model prediction with calibrated probabilities
        prediction = self.model.predict(latest_df)[0]
        prediction_proba = self.model.predict_proba(latest_df)[0]
        model_confidence = prediction_proba[prediction]

        # Uncertainty estimation
        if self.uncertainty_estimator:
            uncertainty_results = self.uncertainty_estimator.predict_with_uncertainty(latest_df)
            total_uncertainty = uncertainty_results["total_uncertainty"][0]
            epistemic_uncertainty = uncertainty_results["epistemic_uncertainty"][0]
        else:
            total_uncertainty = 0.3
            epistemic_uncertainty = 0.1

        # Deep RL prediction (if available)
        rl_prediction = None
        rl_confidence = 0
        if self.rl_agent and self.use_deep_rl:
            try:
                # Prepare state for RL agent
                rl_state = self._prepare_rl_state(current_data)
                rl_action = self.rl_agent.select_action(rl_state, eval_mode=True)
                regime_info = self.rl_agent.detect_market_regime(rl_state)
                uncertainty_info = self.rl_agent.get_uncertainty(rl_state)

                # Convert RL action to direction
                rl_prediction = 1 if rl_action[0] > 0 else 0
                rl_confidence = 1 - uncertainty_info["uncertainty"]

                self.market_regime_history.append(regime_info)
            except Exception as e:
                logger.warning(f"RL prediction failed: {e}")

        # Combine predictions with weighted average
        if rl_prediction is not None:
            # Weight based on confidence and uncertainty
            ml_weight = model_confidence * (1 - total_uncertainty)
            rl_weight = rl_confidence * 0.5  # RL gets less weight initially

            total_weight = ml_weight + rl_weight
            if total_weight > 0:
                combined_prob = (prediction * ml_weight + rl_prediction * rl_weight) / total_weight
                final_prediction = 1 if combined_prob > 0.5 else 0
                final_confidence = (
                    model_confidence * ml_weight + rl_confidence * rl_weight
                ) / total_weight
            else:
                final_prediction = prediction
                final_confidence = model_confidence
        else:
            final_prediction = prediction
            final_confidence = model_confidence

        # Adjust confidence based on uncertainty
        adjusted_confidence = final_confidence * (1 - total_uncertainty * 0.5)

        # Convert current data to DataFrame for technical analysis
        df = pd.DataFrame(current_data)
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "time" in df.columns:
            df = df.set_index("time")

        recent_data = df.tail(20)

        # Technical indicators for strategy refinement
        macd = MACD(close=recent_data["Close"])
        rsi = RSIIndicator(close=recent_data["Close"]).rsi().iloc[-1]
        bb = BollingerBands(close=recent_data["Close"])
        atr = AverageTrueRange(
            high=recent_data["High"], low=recent_data["Low"], close=recent_data["Close"],
        )

        macd_diff = macd.macd_diff().iloc[-1]
        bb_pct = (recent_data["Close"].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (
            bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1] + 1e-10
        )
        volatility = atr.average_true_range().iloc[-1] / recent_data["Close"].iloc[-1]

        # Signal strength from technical indicators
        signal_strength = 0
        if macd_diff > 0:
            signal_strength += 1
        elif macd_diff < 0:
            signal_strength -= 1
        if rsi < 30:
            signal_strength += 1
        elif rsi > 70:
            signal_strength -= 1
        if bb_pct < 0.2:
            signal_strength += 1
        elif bb_pct > 0.8:
            signal_strength -= 1

        # Final direction
        model_direction = "CALL" if final_prediction == 1 else "PUT"

        # Confirmation from technical signals
        if (model_direction == "CALL" and signal_strength > 0) or (
            model_direction == "PUT" and signal_strength < 0
        ):
            adjusted_confidence = min(1.0, adjusted_confidence * 1.15)
        elif (model_direction == "CALL" and signal_strength < -1) or (
            model_direction == "PUT" and signal_strength > 1
        ):
            adjusted_confidence *= 0.85

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
        if total_uncertainty > 0.5:
            duration = int(base_duration * 1.3)  # More time when uncertain
        elif total_uncertainty < 0.2:
            duration = int(base_duration * 0.8)  # Less time when confident
        else:
            duration = base_duration

        duration = max(30, min(300, duration))

        # Create strategy
        self.strategy = {
            "direction": model_direction,
            "confidence": float(adjusted_confidence),
            "original_confidence": float(model_confidence),
            "duration": duration,
            "duration_unit": "s",
            "contract_type": model_direction,
            "volatility": float(volatility),
            "signal_strength": signal_strength,
            "uncertainty": {
                "total": float(total_uncertainty),
                "epistemic": float(epistemic_uncertainty),
                "confidence_adjusted": float(adjusted_confidence),
            },
            "technical_indicators": {
                "macd_diff": float(macd_diff),
                "rsi": float(rsi),
                "bb_pct": float(bb_pct),
            },
            "model_info": {
                "type": "ensemble_stacking_v2",
                "rl_used": rl_prediction is not None,
                "rl_confidence": float(rl_confidence) if rl_prediction is not None else 0,
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Store prediction for analysis
        self.prediction_history.append(
            {
                "prediction": final_prediction,
                "confidence": adjusted_confidence,
                "uncertainty": total_uncertainty,
                "timestamp": datetime.datetime.now(),
            },
        )

        logger.info(
            f"Strategy: {model_direction} @ {adjusted_confidence:.2%} confidence, "
            f"uncertainty: {total_uncertainty:.2%}, duration: {duration}s",
        )

        return self.strategy

    def _prepare_rl_state(self, candle_data):
        """Prepare state for Deep RL agent."""
        df = pd.DataFrame(candle_data)

        # Prepare technical features
        features_df = self._prepare_features(candle_data)
        if len(features_df) < 50:
            # Pad with zeros if not enough data
            technical = np.zeros((50, 20))
        else:
            # Take last 50 rows and first 20 feature columns
            feature_cols = [c for c in features_df.columns if c != "target"][:20]
            technical = features_df[feature_cols].tail(50).values
            if len(technical) < 50:
                pad = np.zeros((50 - len(technical), technical.shape[1]))
                technical = np.vstack([pad, technical])

        # Prepare price data (OHLC)
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        price_cols = ["Open", "High", "Low", "Close"]
        available_cols = [c for c in price_cols if c in df.columns]

        if len(available_cols) == 4:
            price = df[available_cols].tail(50).values
            if len(price) < 50:
                pad = np.zeros((50 - len(price), 4))
                price = np.vstack([pad, price])
            # Normalize
            price = (price - price.mean()) / (price.std() + 1e-10)
        else:
            price = np.zeros((50, 4))

        # Prepare volume data
        if "Volume" in df.columns or "volume" in df.columns:
            vol_col = "Volume" if "Volume" in df.columns else "volume"
            volume = df[vol_col].tail(50).values
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

    def get_market_analysis(self, current_data):
        """Get comprehensive market analysis."""
        features_df = self._prepare_features(current_data)
        if len(features_df) < 5:
            return None

        latest = features_df.iloc[-1]

        analysis = {
            "trend": {
                "direction": "bullish" if latest.get("macd_diff", 0) > 0 else "bearish",
                "strength": abs(latest.get("adx", 0)) / 100,
                "macd_signal": "buy" if latest.get("macd_diff", 0) > 0 else "sell",
            },
            "momentum": {
                "rsi": latest.get("rsi", 50),
                "rsi_signal": "oversold"
                if latest.get("rsi", 50) < 30
                else "overbought"
                if latest.get("rsi", 50) > 70
                else "neutral",
                "stochastic": latest.get("stoch_k", 50),
                "williams_r": latest.get("williams_r", -50),
            },
            "volatility": {
                "bb_width": latest.get("bb_width", 0),
                "atr_ratio": latest.get("atr_ratio", 0),
                "regime": "high"
                if latest.get("volatility_ratio", 1) > 1.5
                else "low"
                if latest.get("volatility_ratio", 1) < 0.7
                else "normal",
            },
            "volume": {
                "trend": "increasing" if latest.get("volume_ratio", 1) > 1 else "decreasing",
                "mfi": latest.get("mfi", 50),
            },
        }

        if self.rl_agent and self.use_deep_rl:
            try:
                rl_state = self._prepare_rl_state(current_data)
                regime_info = self.rl_agent.detect_market_regime(rl_state)
                analysis["market_regime"] = regime_info
            except Exception as e:
                logger.warning(f"Could not get market regime: {e}")

        return analysis

    def update_strategy_from_results(self, trade_results):
        """Update strategy based on trade results with adaptive learning."""
        if not trade_results or self.model is None:
            return self.strategy

        logger.info("Updating strategy based on trade results...")

        # Calculate metrics
        wins = sum(1 for trade in trade_results if trade.get("profit", 0) > 0)
        total = len(trade_results)
        win_rate = wins / total if total > 0 else 0

        profits = [trade.get("profit", 0) for trade in trade_results if trade.get("profit", 0) > 0]
        losses = [
            abs(trade.get("profit", 0)) for trade in trade_results if trade.get("profit", 0) < 0
        ]

        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        risk_reward_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

        # Update performance metrics
        self.performance_metrics.update(
            {
                "win_rate": win_rate,
                "trades_count": self.performance_metrics.get("trades_count", 0) + total,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "risk_reward_ratio": risk_reward_ratio,
            },
        )

        # Check consecutive losses
        consecutive_losses = 0
        for trade in reversed(trade_results):
            if trade.get("profit", 0) <= 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            logger.info(
                f"Detected {consecutive_losses} consecutive losses. Forcing strategy recalculation...",
            )
            self.strategy = None
            return None

        if self.strategy:
            self.strategy["performance"] = {
                "win_rate": win_rate,
                "risk_reward_ratio": risk_reward_ratio,
                "consecutive_losses": consecutive_losses,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
            }

        return self.strategy

    def get_model_summary(self):
        """Get summary of current model state."""
        return {
            "trained": self.model is not None,
            "last_training": self.last_training_time.isoformat()
            if self.last_training_time
            else None,
            "performance": self.performance_metrics,
            "deep_rl_enabled": self.use_deep_rl,
            "enhanced_features_enabled": self.use_enhanced_features,
            "prediction_history_length": len(self.prediction_history),
            "market_regime_history_length": len(self.market_regime_history),
        }

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime
import joblib
import os
import logging
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('trading_ai')

class TradingAI:
    """AI system for market analysis and trading strategy generation."""

    def __init__(self, model_dir='models'):
        """Initialize the TradingAI system."""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.last_training_time = None
        self.strategy = None
        self.performance_metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'win_rate': 0,
            'trades_count': 0
        }

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def _prepare_features(self, df):
        """
        Prepare features for model training from raw candle data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators as features
        """
        # Ensure we have a DataFrame with the right columns
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        # Standardize column names
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'epoch': 'epoch'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Set time as index if it exists
        if 'time' in df.columns:
            df = df.set_index('time')

        # Ensure we have enough data
        if len(df) < 30:
            logger.warning(f"Not enough data for feature calculation: {len(df)} rows")
            return pd.DataFrame()

        # Calculate technical indicators
        features = pd.DataFrame(index=df.index)

        # Price-based indicators
        features['price_change'] = df['Close'].pct_change()
        features['price_change_2'] = df['Close'].pct_change(2)
        features['price_change_5'] = df['Close'].pct_change(5)

        # Volatility indicators
        features['high_low_diff'] = (df['High'] - df['Low']) / df['Open']
        features['close_open_diff'] = (df['Close'] - df['Open']) / df['Open']

        # TA-Lib indicators
        # Trend indicators
        macd = MACD(close=df['Close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()

        # Moving averages using TA
        sma5 = SMAIndicator(close=df['Close'], window=5)
        sma10 = SMAIndicator(close=df['Close'], window=10)
        sma20 = SMAIndicator(close=df['Close'], window=20)
        ema5 = EMAIndicator(close=df['Close'], window=5)
        ema10 = EMAIndicator(close=df['Close'], window=10)

        features['sma_5'] = sma5.sma_indicator()
        features['sma_10'] = sma10.sma_indicator()
        features['sma_20'] = sma20.sma_indicator()
        features['ema_5'] = ema5.ema_indicator()
        features['ema_10'] = ema10.ema_indicator()

        # MA crossovers
        features['sma_5_10_diff'] = features['sma_5'] - features['sma_10']
        features['sma_10_20_diff'] = features['sma_10'] - features['sma_20']
        features['ema_5_10_diff'] = features['ema_5'] - features['ema_10']

        # Momentum indicators
        rsi = RSIIndicator(close=df['Close'])
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])

        features['rsi'] = rsi.rsi()
        features['stoch_k'] = stoch.stoch()
        features['stoch_d'] = stoch.stoch_signal()

        # Volatility indicators
        bb = BollingerBands(close=df['Close'])
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])

        features['bb_high'] = bb.bollinger_hband()
        features['bb_low'] = bb.bollinger_lband()
        features['bb_width'] = (features['bb_high'] - features['bb_low']) / df['Close']
        features['bb_pct'] = (df['Close'] - features['bb_low']) / (features['bb_high'] - features['bb_low'])
        features['atr'] = atr.average_true_range()

        # Volume indicators
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
            vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])

            features['obv'] = obv.on_balance_volume()
            features['vwap'] = vwap.volume_weighted_average_price()
            features['volume_change'] = df['Volume'].pct_change()
            features['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
            features['volume_ratio'] = df['Volume'] / features['volume_ma_5']

        # Target: Price direction (1 if price goes up in next period, 0 otherwise)
        features['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Drop NaN values
        features = features.dropna()

        return features

    def _calculate_rsi(self, prices, window=14):
        """Calculate the Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def train_model(self, historical_data):
        """
        Train a machine learning model on historical market data.

        Args:
            historical_data: List of candles or DataFrame with OHLCV data

        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")

        # Prepare features
        features_df = self._prepare_features(historical_data)

        if len(features_df) < 100:
            logger.warning(f"Insufficient data for training: {len(features_df)} samples")
            return None

        # Split features and target
        X = features_df.drop('target', axis=1)
        y = features_df['target']

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Get symbol name from the data if available
        symbol = None
        if isinstance(historical_data, list) and len(historical_data) > 0:
            if isinstance(historical_data[0], dict) and 'symbol' in historical_data[0]:
                symbol = historical_data[0]['symbol']

        # Train multiple models for comparison
        models = {
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1))
            ]),
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10))
            ]),
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, C=0.1))
            ])
        }

        best_model = None
        best_precision = 0
        best_model_name = None
        model_metrics = {}

        # Train and evaluate each model
        for model_name, model_pipeline in models.items():
            logger.info(f"Training {model_name} model...")

            # Train the model
            model_pipeline.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = model_pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            # Cross-validation score
            cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='precision')
            cv_precision = cv_scores.mean()

            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)

            # Store metrics
            model_metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_precision': cv_precision,
                'confusion_matrix': cm.tolist()
            }

            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                       f"Recall: {recall:.4f}, F1: {f1:.4f}, CV Precision: {cv_precision:.4f}")

            # Save model to disk
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{symbol}_{model_name}_{timestamp}.joblib" if symbol else f"trading_model_{model_name}_{timestamp}.joblib"
            model_path = os.path.join(self.model_dir, model_filename)
            joblib.dump(model_pipeline, model_path)

            # Track the best model based on precision
            if precision > best_precision:
                best_precision = precision
                best_model = model_pipeline
                best_model_name = model_name

        # Use the best model
        if best_model:
            self.model = best_model
            self.last_training_time = datetime.datetime.now()

            # Save performance metrics
            self.performance_metrics = {
                'accuracy': model_metrics[best_model_name]['accuracy'],
                'precision': model_metrics[best_model_name]['precision'],
                'recall': model_metrics[best_model_name]['recall'],
                'f1': model_metrics[best_model_name]['f1'],
                'cv_precision': model_metrics[best_model_name]['cv_precision'],
                'win_rate': model_metrics[best_model_name]['precision'],  # Initial win rate estimate
                'trades_count': 0,
                'best_model': best_model_name,
                'all_models': model_metrics
            }

            logger.info(f"Selected best model: {best_model_name} with precision: {best_precision:.4f}")
        else:
            logger.error("Failed to train any models successfully")
            return None

        return self.performance_metrics

    def generate_strategy(self, current_data, recalculate=False):
        """
        Generate a trading strategy based on the trained model and current market data.

        Args:
            current_data: Recent candle data
            recalculate: Whether to force strategy recalculation

        Returns:
            Dictionary with strategy parameters
        """
        if self.model is None:
            logger.warning("No trained model available. Cannot generate strategy.")
            return None

        if self.strategy is not None and not recalculate:
            logger.info("Using existing strategy")
            return self.strategy

        logger.info("Generating new trading strategy...")

        # Prepare features from current data
        features_df = self._prepare_features(current_data)

        if len(features_df) < 5:
            logger.warning(f"Insufficient data for strategy generation: {len(features_df)} samples")
            return None

        # Get the most recent data point
        latest_features = features_df.iloc[-1:].drop('target', axis=1)

        # Predict price direction
        prediction = self.model.predict(latest_features)[0]
        prediction_proba = self.model.predict_proba(latest_features)[0]
        confidence = prediction_proba[prediction]

        # Convert current_data to DataFrame for analysis
        df = pd.DataFrame(current_data)
        if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        # Standardize column names
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'epoch': 'epoch'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if 'time' in df.columns:
            df = df.set_index('time')

        # Get recent data for analysis
        recent_data = df.tail(20)

        # Calculate technical indicators for strategy refinement
        # Trend indicators
        macd = MACD(close=recent_data['Close'])
        macd_value = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        macd_diff = macd.macd_diff().iloc[-1]

        # RSI
        rsi = RSIIndicator(close=recent_data['Close']).rsi().iloc[-1]

        # Bollinger Bands
        bb = BollingerBands(close=recent_data['Close'])
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]
        bb_pct = (recent_data['Close'].iloc[-1] - bb_low) / (bb_high - bb_low)

        # Volatility
        atr = AverageTrueRange(high=recent_data['High'], low=recent_data['Low'], close=recent_data['Close'])
        volatility = atr.average_true_range().iloc[-1] / recent_data['Close'].iloc[-1]

        # Determine trade direction based on model prediction and technical indicators
        model_direction = "CALL" if prediction == 1 else "PUT"

        # Technical signal strength (additional confirmation)
        signal_strength = 0

        # MACD signals
        if macd_diff > 0 and macd_value > 0:
            signal_strength += 1  # Bullish
        elif macd_diff < 0 and macd_value < 0:
            signal_strength -= 1  # Bearish

        # RSI signals
        if rsi < 30:
            signal_strength += 1  # Oversold, potential upside
        elif rsi > 70:
            signal_strength -= 1  # Overbought, potential downside

        # Bollinger Band signals
        if bb_pct < 0.2:
            signal_strength += 1  # Near lower band, potential upside
        elif bb_pct > 0.8:
            signal_strength -= 1  # Near upper band, potential downside

        # Adjust confidence based on technical signal alignment with model prediction
        adjusted_confidence = confidence
        if (model_direction == "CALL" and signal_strength > 0) or (model_direction == "PUT" and signal_strength < 0):
            # Technical signals confirm model prediction
            adjusted_confidence = min(1.0, confidence * 1.2)
        elif (model_direction == "CALL" and signal_strength < 0) or (model_direction == "PUT" and signal_strength > 0):
            # Technical signals contradict model prediction
            adjusted_confidence = confidence * 0.8

        # Final direction decision
        # If confidence is too low after adjustment, consider reversing the direction
        if adjusted_confidence < 0.4 and abs(signal_strength) >= 2:
            direction = "PUT" if signal_strength > 0 else "CALL"  # Reverse the direction based on strong technical signals
            logger.info(f"Reversing direction due to strong technical signals: {direction}")
        else:
            direction = model_direction

        # Determine trade duration based on volatility and market conditions
        if volatility > 0.015:  # Very high volatility
            duration = 30  # 30 seconds
        elif volatility > 0.01:  # High volatility
            duration = 60  # 1 minute
        elif volatility > 0.005:  # Medium volatility
            duration = 120  # 2 minutes
        else:  # Low volatility
            duration = 180  # 3 minutes

        # Adjust duration based on signal strength
        if abs(signal_strength) >= 2:
            # Strong signal, can use shorter duration
            duration = max(30, int(duration * 0.8))
        elif abs(signal_strength) == 0:
            # Neutral signals, use longer duration
            duration = min(300, int(duration * 1.2))

        # Create comprehensive strategy
        self.strategy = {
            'direction': direction,
            'model_direction': model_direction,
            'confidence': float(adjusted_confidence),
            'original_confidence': float(confidence),
            'duration': duration,
            'duration_unit': 's',
            'contract_type': 'CALL' if direction == 'CALL' else 'PUT',
            'volatility': float(volatility),
            'signal_strength': signal_strength,
            'technical_indicators': {
                'macd': float(macd_value),
                'macd_signal': float(macd_signal),
                'macd_diff': float(macd_diff),
                'rsi': float(rsi),
                'bb_pct': float(bb_pct)
            },
            'timestamp': datetime.datetime.now().isoformat()
        }

        logger.info(f"Strategy generated: {direction} with {adjusted_confidence:.4f} confidence (original: {confidence:.4f}), "
                   f"duration: {duration}s, signal strength: {signal_strength}")

        return self.strategy

    def update_strategy_from_results(self, trade_results):
        """
        Update the trading strategy based on recent trade results.

        Args:
            trade_results: List of recent trade results

        Returns:
            Updated strategy dictionary
        """
        if not trade_results or self.model is None:
            return self.strategy

        logger.info("Updating strategy based on trade results...")

        # Calculate win rate and other metrics from recent trades
        wins = sum(1 for trade in trade_results if trade.get('profit', 0) > 0)
        total = len(trade_results)
        win_rate = wins / total if total > 0 else 0

        # Calculate average profit and loss
        profits = [trade.get('profit', 0) for trade in trade_results if trade.get('profit', 0) > 0]
        losses = [abs(trade.get('profit', 0)) for trade in trade_results if trade.get('profit', 0) < 0]

        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Calculate risk-reward ratio
        risk_reward_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

        # Update performance metrics
        self.performance_metrics['win_rate'] = win_rate
        self.performance_metrics['trades_count'] += total
        self.performance_metrics['avg_profit'] = avg_profit
        self.performance_metrics['avg_loss'] = avg_loss
        self.performance_metrics['risk_reward_ratio'] = risk_reward_ratio

        # Analyze trade results by direction
        call_trades = [t for t in trade_results if t.get('contract_info', {}).get('contract_type', '') == 'CALL']
        put_trades = [t for t in trade_results if t.get('contract_info', {}).get('contract_type', '') == 'PUT']

        call_win_rate = sum(1 for t in call_trades if t.get('profit', 0) > 0) / len(call_trades) if call_trades else 0
        put_win_rate = sum(1 for t in put_trades if t.get('profit', 0) > 0) / len(put_trades) if put_trades else 0

        # Analyze trades by duration
        short_trades = [t for t in trade_results if t.get('contract_info', {}).get('duration', 0) <= 60]
        medium_trades = [t for t in trade_results if 60 < t.get('contract_info', {}).get('duration', 0) <= 180]
        long_trades = [t for t in trade_results if t.get('contract_info', {}).get('duration', 0) > 180]

        short_win_rate = sum(1 for t in short_trades if t.get('profit', 0) > 0) / len(short_trades) if short_trades else 0
        medium_win_rate = sum(1 for t in medium_trades if t.get('profit', 0) > 0) / len(medium_trades) if medium_trades else 0
        long_win_rate = sum(1 for t in long_trades if t.get('profit', 0) > 0) / len(long_trades) if long_trades else 0

        # Check if we need to recalculate the strategy
        consecutive_losses = 0
        for trade in reversed(trade_results):
            if trade.get('profit', 0) <= 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            logger.info(f"Detected {consecutive_losses} consecutive losses. Recalculating strategy...")
            return None  # Force strategy recalculation

        # Adjust strategy based on performance analysis
        if self.strategy:
            # Original strategy parameters
            original_direction = self.strategy.get('direction', 'CALL')
            original_duration = self.strategy.get('duration', 60)

            # Determine if we should favor a particular direction
            if call_win_rate > 0.6 and call_win_rate > put_win_rate + 0.2:
                # Strongly favor CALL
                direction_bias = 0.3  # Increase confidence for CALL
            elif put_win_rate > 0.6 and put_win_rate > call_win_rate + 0.2:
                # Strongly favor PUT
                direction_bias = -0.3  # Increase confidence for PUT
            elif call_win_rate > put_win_rate + 0.1:
                # Slightly favor CALL
                direction_bias = 0.1
            elif put_win_rate > call_win_rate + 0.1:
                # Slightly favor PUT
                direction_bias = -0.1
            else:
                # No clear bias
                direction_bias = 0

            # Store direction bias for future strategy generation
            self.strategy['direction_bias'] = direction_bias

            # Determine optimal duration based on win rates
            if short_win_rate > medium_win_rate and short_win_rate > long_win_rate:
                optimal_duration = min(60, original_duration)
            elif medium_win_rate > short_win_rate and medium_win_rate > long_win_rate:
                optimal_duration = max(60, min(180, original_duration))
            elif long_win_rate > 0.5:
                optimal_duration = max(180, original_duration)
            else:
                # Adjust duration based on overall win rate
                if win_rate < 0.4:
                    # Increase duration for more time to reach target
                    optimal_duration = min(300, original_duration * 1.5)
                elif win_rate > 0.6:
                    # Decrease duration for faster trades
                    optimal_duration = max(30, original_duration * 0.8)
                else:
                    optimal_duration = original_duration

            # Update strategy with new duration
            self.strategy['duration'] = optimal_duration

            # Update strategy with performance metrics
            self.strategy['performance'] = {
                'win_rate': win_rate,
                'call_win_rate': call_win_rate,
                'put_win_rate': put_win_rate,
                'short_win_rate': short_win_rate,
                'medium_win_rate': medium_win_rate,
                'long_win_rate': long_win_rate,
                'risk_reward_ratio': risk_reward_ratio,
                'consecutive_losses': consecutive_losses
            }

            logger.info(f"Strategy updated. Direction bias: {direction_bias:.2f}, "
                       f"New duration: {optimal_duration}s, Win rate: {win_rate:.4f}")

        return self.strategy

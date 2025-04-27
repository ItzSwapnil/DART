import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import datetime
import joblib
import os
import logging

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
        
        # Calculate technical indicators
        features = pd.DataFrame(index=df.index)
        
        # Price-based indicators
        features['price_change'] = df['Close'].pct_change()
        features['price_change_2'] = df['Close'].pct_change(2)
        features['price_change_5'] = df['Close'].pct_change(5)
        
        # Volatility indicators
        features['high_low_diff'] = (df['High'] - df['Low']) / df['Open']
        features['close_open_diff'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        features['ma_5'] = df['Close'].rolling(window=5).mean()
        features['ma_10'] = df['Close'].rolling(window=10).mean()
        features['ma_20'] = df['Close'].rolling(window=20).mean()
        
        # MA crossovers
        features['ma_5_10_diff'] = features['ma_5'] - features['ma_10']
        features['ma_10_20_diff'] = features['ma_10'] - features['ma_20']
        
        # Momentum indicators
        features['rsi_5'] = self._calculate_rsi(df['Close'], window=5)
        features['rsi_14'] = self._calculate_rsi(df['Close'], window=14)
        
        # Volume indicators
        if 'Volume' in df.columns:
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
        
        # Create and train the model pipeline
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model_pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        # Save model metrics
        self.performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'win_rate': precision,  # Initial win rate estimate based on precision
            'trades_count': 0
        }
        
        # Save the model
        self.model = model_pipeline
        self.last_training_time = datetime.datetime.now()
        
        # Save model to disk
        model_path = os.path.join(self.model_dir, f'trading_model_{self.last_training_time.strftime("%Y%m%d_%H%M%S")}.joblib')
        joblib.dump(model_pipeline, model_path)
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
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
        
        # Determine optimal trade parameters based on recent volatility
        recent_data = pd.DataFrame(current_data).tail(20)
        
        if 'time' in recent_data.columns:
            recent_data['time'] = pd.to_datetime(recent_data['time'])
            recent_data = recent_data.set_index('time')
            
        # Standardize column names
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        recent_data = recent_data.rename(columns={k: v for k, v in rename_map.items() if k in recent_data.columns})
        
        # Calculate volatility
        if 'High' in recent_data.columns and 'Low' in recent_data.columns:
            volatility = ((recent_data['High'] - recent_data['Low']) / recent_data['Open']).mean()
        else:
            volatility = recent_data['Close'].pct_change().std()
        
        # Determine trade direction
        direction = "CALL" if prediction == 1 else "PUT"
        
        # Determine trade duration based on volatility
        if volatility > 0.01:  # High volatility
            duration = 60  # 1 minute
        elif volatility > 0.005:  # Medium volatility
            duration = 120  # 2 minutes
        else:  # Low volatility
            duration = 180  # 3 minutes
            
        # Create strategy
        self.strategy = {
            'direction': direction,
            'confidence': float(confidence),
            'duration': duration,
            'duration_unit': 's',
            'contract_type': 'CALL' if direction == 'CALL' else 'PUT',
            'volatility': float(volatility),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Strategy generated: {direction} with {confidence:.4f} confidence, duration: {duration}s")
        
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
        
        # Calculate win rate from recent trades
        wins = sum(1 for trade in trade_results if trade.get('profit', 0) > 0)
        total = len(trade_results)
        win_rate = wins / total if total > 0 else 0
        
        # Update performance metrics
        self.performance_metrics['win_rate'] = win_rate
        self.performance_metrics['trades_count'] += total
        
        # Check if we need to recalculate the strategy
        consecutive_losses = sum(1 for trade in trade_results if trade.get('profit', 0) <= 0)
        
        if consecutive_losses >= 3:
            logger.info(f"Detected {consecutive_losses} consecutive losses. Recalculating strategy...")
            return None  # Force strategy recalculation
            
        # Adjust duration based on win rate
        if self.strategy:
            if win_rate < 0.4:
                # Increase duration for more time to reach target
                self.strategy['duration'] = min(300, self.strategy['duration'] * 1.5)
            elif win_rate > 0.6:
                # Decrease duration for faster trades
                self.strategy['duration'] = max(30, self.strategy['duration'] * 0.8)
                
            logger.info(f"Strategy updated. New duration: {self.strategy['duration']}s, Win rate: {win_rate:.4f}")
            
        return self.strategy
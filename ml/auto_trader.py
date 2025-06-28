import asyncio
import datetime
import json
import logging
import os
import threading

import numpy as np
import pandas as pd

from api.deriv_client import DerivClient
from config.settings import (
    CONFIDENCE_THRESHOLD,
    DEEP_RL_CONFIG,
    MAX_DAILY_LOSS,
    MODEL_UPDATE_FREQUENCY,
    RISK_MANAGEMENT_CONFIG,
    TRADE_AMOUNT,
    TRAINING_DAYS,
)
from ml.trading_ai import TradingAI

# Configure logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("auto_trader")

# Import new components with graceful fallback
try:
    from ml.risk_manager import RiskManager

    RISK_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Advanced risk manager not available - using basic risk controls")
    RISK_MANAGER_AVAILABLE = False

try:
    from ml.deep_rl_agent import DeepRLAgent

    DEEP_RL_AVAILABLE = True
except ImportError:
    logger.warning("Deep RL agent not available - using traditional ML models")
    DEEP_RL_AVAILABLE = False


class AutoTrader:
    """
    Automated trading manager that coordinates between the TradingAI and DerivClient
    to execute trades based on generated strategies.
    Enhanced with advanced risk management and optional deep RL capabilities.
    """

    def __init__(self, client: DerivClient, trading_ai: TradingAI):
        """Initialize the AutoTrader with a DerivClient and TradingAI instance."""
        self.client = client
        self.trading_ai = trading_ai
        self.is_running = False
        self.trading_thread = None
        self.loop = None
        self.current_symbol = None
        self.current_granularity = None

        # Initialize advanced components if available
        self.risk_manager = None
        self.deep_rl_agent = None

        if RISK_MANAGER_AVAILABLE and RISK_MANAGEMENT_CONFIG.get("enabled", False):
            try:
                self.risk_manager = RiskManager(
                    initial_balance=self.get_current_balance(), config=RISK_MANAGEMENT_CONFIG,
                )
                logger.info("Advanced risk manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize risk manager: {e}")

        if DEEP_RL_AVAILABLE and DEEP_RL_CONFIG.get("enabled", False):
            try:
                self.deep_rl_agent = DeepRLAgent(
                    state_dim=DEEP_RL_CONFIG.get("state_dim", 50),
                    action_dim=DEEP_RL_CONFIG.get("action_dim", 3),
                    config=DEEP_RL_CONFIG,
                )
                logger.info("Deep RL agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize deep RL agent: {e}")

        # Risk management
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.session_start_time = datetime.datetime.now()
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0

        # Performance tracking
        self.trade_count = 0
        self.successful_trades = 0
        self.trade_history = []
        self.performance_by_hour = {hour: {"trades": 0, "wins": 0} for hour in range(24)}
        self.performance_by_market = {}

        # Model management
        self.last_model_training = None
        self.model_performance_history = []

        # Adaptive trade sizing
        self.base_trade_amount = TRADE_AMOUNT
        self.current_trade_amount = TRADE_AMOUNT
        self.trade_size_multiplier = 1.0

        # Status and callbacks
        self.status_callbacks = []
        self.trading_paused = False
        self.pause_reason = None

        # Create logs directory
        self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    def register_status_callback(self, callback):
        """Register a callback function to receive status updates."""
        self.status_callbacks.append(callback)

    def _notify_status(self, status_data):
        """Notify all registered callbacks with status updates."""
        for callback in self.status_callbacks:
            try:
                callback(status_data)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    async def check_api_connection(self):
        """
        Check API connection and circuit breaker state before trading.

        Returns:
            True if API is available, False otherwise
        """
        try:
            # Check if the client has a circuit breaker and if it's open
            if hasattr(self.client, '_circuit_breaker'):
                from utils.api_utils import CircuitState
                cb_state = self.client._circuit_breaker.state
                if cb_state != CircuitState.CLOSED:
                    logger.warning(
                        f"API circuit breaker is {cb_state.value}. "
                        "Waiting for recovery...",
                    )
                    self._notify_status({
                        "status": "warning",
                        "message": f"API circuit breaker {cb_state.value}. "
                                   "Waiting for recovery...",
                    })
                    return False

            # Check actual connection
            is_connected = await self.client.check_connection()

            if not is_connected:
                logger.warning("API connection check failed")
                self._notify_status({
                    "status": "warning",
                    "message": "API connection lost. Attempting reconnection...",
                })

                # Try to re-establish connection
                account_info = await self.client.get_account_info()
                if account_info:
                    logger.info("API connection re-established")
                    self._notify_status({
                        "status": "info",
                        "message": "API connection re-established",
                    })
                    return True
                else:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking API connection: {e}")
            self._notify_status({
                "status": "error",
                "message": f"Connection check failed: {str(e)}",
            })
            return False

    async def train_model(self, symbol, granularity):
        """Train the AI model on historical data for the specified symbol and granularity."""
        self._notify_status({"status": "training", "message": "Collecting historical data..."})

        # Get historical data for training
        historical_data = await self.client.get_historical_data(
            symbol, granularity, days=TRAINING_DAYS,
        )

        if not historical_data or len(historical_data) < 100:
            message = f"Insufficient historical data: {len(historical_data) if historical_data else 0} candles"
            logger.warning(message)
            self._notify_status({"status": "error", "message": message})
            return False

        self._notify_status(
            {
                "status": "training",
                "message": f"Training model on {len(historical_data)} candles...",
            },
        )

        # Train the model
        metrics = self.trading_ai.train_model(historical_data)

        if not metrics:
            message = "Model training failed"
            logger.error(message)
            self._notify_status({"status": "error", "message": message})
            return False

        self.last_model_training = datetime.datetime.now()

        self._notify_status(
            {"status": "trained", "message": "Model training completed", "metrics": metrics},
        )

        return True

    async def generate_strategy(self, symbol, granularity, recalculate=False):
        """Generate a trading strategy for the specified symbol and granularity."""
        # Get recent data for strategy generation
        recent_data = await self.client.get_candles(symbol, granularity, count=50)

        if not recent_data or len(recent_data) < 20:
            message = f"Insufficient recent data: {len(recent_data) if recent_data else 0} candles"
            logger.warning(message)
            self._notify_status({"status": "error", "message": message})
            return None

        # Generate base strategy using traditional ML
        strategy = self.trading_ai.generate_strategy(recent_data, recalculate=recalculate)

        if not strategy:
            message = "Failed to generate strategy. Please train the model first."
            logger.warning(message)
            self._notify_status({"status": "error", "message": message})
            return None

        # Enhance strategy with deep RL if available
        if self.deep_rl_agent:
            try:
                # Convert recent data to state representation
                state = self._prepare_rl_state(recent_data)

                # Get RL action and confidence
                rl_action, rl_confidence = self.deep_rl_agent.select_action(
                    state, deterministic=True,
                )

                # Combine ML and RL predictions
                strategy = self._combine_predictions(strategy, rl_action, rl_confidence)

                logger.info(
                    f"Enhanced strategy with Deep RL: {strategy['direction']} (confidence: {strategy['confidence']:.2f})",
                )

            except Exception as e:
                logger.warning(f"Deep RL enhancement failed: {e}, using base strategy")

        # Apply risk management constraints
        if self.risk_manager:
            try:
                # Check risk constraints before trading
                risk_check = self.risk_manager.check_trade_risk(
                    symbol=symbol,
                    direction=strategy["direction"],
                    amount=self.current_trade_amount,
                    confidence=strategy["confidence"],
                )

                if not risk_check["allowed"]:
                    strategy["risk_blocked"] = True
                    strategy["risk_reason"] = risk_check["reason"]
                    logger.warning(f"Trade blocked by risk management: {risk_check['reason']}")
                else:
                    # Adjust position size based on risk management
                    recommended_size = self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        direction=strategy["direction"],
                        volatility=strategy.get("volatility", 0.02),
                        confidence=strategy["confidence"],
                    )

                    strategy["recommended_amount"] = recommended_size
                    strategy["risk_approved"] = True

            except Exception as e:
                logger.warning(f"Risk management check failed: {e}, proceeding with base strategy")

        self._notify_status(
            {
                "status": "strategy",
                "message": f"Strategy generated: {strategy['direction']} with {strategy['confidence']:.2f} confidence",
                "strategy": strategy,
            },
        )

        return strategy

    def _prepare_rl_state(self, candle_data):
        """Prepare state representation for deep RL agent."""
        try:
            # Use the trading AI's feature extraction if available
            if hasattr(self.trading_ai, "feature_extractor") and self.trading_ai.feature_extractor:
                features = self.trading_ai.feature_extractor.extract_features(candle_data)
                return features[-1]  # Get latest feature vector
            else:
                # Simple state representation using basic technical indicators
                df = pd.DataFrame(candle_data)

                # Basic features
                close_prices = df["close"].values
                returns = np.diff(close_prices) / close_prices[:-1]

                # Simple moving averages
                sma_5 = np.mean(close_prices[-5:])
                sma_20 = np.mean(close_prices[-20:])

                # Volatility
                volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02

                # Price position relative to recent range
                recent_high = np.max(close_prices[-20:])
                recent_low = np.min(close_prices[-20:])
                price_position = (
                    (close_prices[-1] - recent_low) / (recent_high - recent_low)
                    if recent_high != recent_low
                    else 0.5
                )

                # Combine features
                state = np.array(
                    [
                        returns[-1] if len(returns) > 0 else 0,  # Last return
                        (close_prices[-1] - sma_5) / sma_5 if sma_5 > 0 else 0,  # Price vs SMA5
                        (close_prices[-1] - sma_20) / sma_20 if sma_20 > 0 else 0,  # Price vs SMA20
                        volatility,  # Volatility
                        price_position,  # Price position in range
                    ],
                )

                # Pad to expected state dimension
                target_dim = self.deep_rl_agent.state_dim
                if len(state) < target_dim:
                    state = np.concatenate([state, np.zeros(target_dim - len(state))])
                elif len(state) > target_dim:
                    state = state[:target_dim]

                return state

        except Exception as e:
            logger.error(f"Error preparing RL state: {e}")
            # Return default state
            return np.zeros(self.deep_rl_agent.state_dim)

    def _combine_predictions(self, ml_strategy, rl_action, rl_confidence):
        """Combine ML and RL predictions into final strategy."""
        try:
            # Map RL action to direction
            rl_direction_map = {0: "CALL", 1: "PUT", 2: "HOLD"}
            rl_direction = rl_direction_map.get(rl_action, "HOLD")

            # Combine confidences (weighted average)
            ml_weight = 0.4  # Weight for traditional ML
            rl_weight = 0.6  # Weight for deep RL

            if ml_strategy["direction"] == rl_direction:
                # Predictions agree - boost confidence
                combined_confidence = min(
                    ml_weight * ml_strategy["confidence"] + rl_weight * rl_confidence + 0.1, 1.0,
                )
                final_direction = ml_strategy["direction"]
            else:
                # Predictions disagree - use higher confidence or fallback to HOLD
                if rl_confidence > ml_strategy["confidence"]:
                    final_direction = rl_direction
                    combined_confidence = (
                        rl_confidence * 0.8
                    )  # Reduce confidence due to disagreement
                else:
                    final_direction = ml_strategy["direction"]
                    combined_confidence = ml_strategy["confidence"] * 0.8

                # If confidence is too low due to disagreement, recommend HOLD
                if combined_confidence < 0.6:
                    final_direction = "HOLD"
                    combined_confidence = 0.5

            # Update strategy
            enhanced_strategy = ml_strategy.copy()
            enhanced_strategy["direction"] = final_direction
            enhanced_strategy["confidence"] = combined_confidence
            enhanced_strategy["ml_prediction"] = {
                "direction": ml_strategy["direction"],
                "confidence": ml_strategy["confidence"],
            }
            enhanced_strategy["rl_prediction"] = {
                "direction": rl_direction,
                "confidence": rl_confidence,
                "action": rl_action,
            }
            enhanced_strategy["prediction_method"] = "combined_ml_rl"

            return enhanced_strategy

        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return ml_strategy

    async def execute_trade(self, symbol, strategy, amount=None, price=None, use_ai_price=False):
        """
        Execute a trade based on the generated strategy.

        Args:
            symbol: Market symbol to trade
            strategy: Trading strategy dictionary
            amount: Trade amount (uses current_trade_amount if None)
            price: Manual price setting (optional)
            use_ai_price: Whether to let AI determine the optimal price

        Returns:
            Contract ID if successful, None otherwise
        """
        # Use provided amount or current trade amount
        trade_amount = amount if amount is not None else self.current_trade_amount

        # Check if trading is paused
        if self.trading_paused:
            logger.warning(f"Trading is paused: {self.pause_reason}")
            self._notify_status({"status": "paused", "message": self.pause_reason})
            return None

        # Check if we've exceeded the daily loss limit
        if self.daily_loss >= MAX_DAILY_LOSS:
            message = f"Daily loss limit reached: ${self.daily_loss:.2f}"
            logger.warning(message)
            self._notify_status({"status": "stopped", "message": message})
            self.trading_paused = True
            self.pause_reason = message
            return None

        # Check for excessive drawdown
        if self.peak_balance > 0:
            drawdown_percentage = self.max_drawdown / self.peak_balance
            if drawdown_percentage > 0.3:  # 30% drawdown
                message = f"Maximum drawdown threshold reached: {drawdown_percentage:.1%}"
                logger.warning(message)
                self._notify_status({"status": "stopped", "message": message})
                self.trading_paused = True
                self.pause_reason = message
                return None

        # Execute the trade
        contract_type = strategy["contract_type"]
        duration = strategy["duration"]
        duration_unit = strategy["duration_unit"]

        # Check if the duration is supported for this market
        try:
            available_durations = await self.client.get_available_durations(symbol, contract_type)

            if available_durations and duration not in available_durations:
                original_duration = duration

                # Find the closest supported duration
                if available_durations:
                    # Find the closest duration (prefer longer over shorter if equidistant)
                    closest_duration = min(
                        available_durations, key=lambda x: (abs(x - duration), -x),
                    )
                    duration = closest_duration

                    logger.info(
                        f"Adjusted duration from {original_duration}s to {duration}s for {symbol} (supported durations: {available_durations})",
                    )

                    # Update the strategy with the new duration
                    strategy["duration"] = duration

                    # Notify the user about the duration adjustment
                    self._notify_status(
                        {
                            "status": "info",
                            "message": f"Adjusted duration from {original_duration}s to {duration}s for {symbol}",
                        },
                    )
        except Exception as e:
            logger.error(f"Error checking available durations: {e}")
            # Continue with the original duration as a fallback

        # Determine price to use
        trade_price = price

        # If AI price management is enabled, calculate optimal price
        if use_ai_price and not price:
            # Get recent market data to determine optimal price
            try:
                recent_candles = await self.client.get_candles(
                    symbol, 60, count=10,
                )  # Get recent 1-minute candles
                if recent_candles and len(recent_candles) > 0:
                    # For CALL (buy), use a price slightly below current market price
                    # For PUT (sell), use a price slightly above current market price
                    latest_price = float(recent_candles[-1]["close"])

                    if contract_type == "CALL":
                        # For buying, try to get a better entry price (0.1% below market)
                        trade_price = round(latest_price * 0.999, 4)
                        logger.info(
                            f"AI price management: Setting CALL price to {trade_price} (market: {latest_price})",
                        )
                    else:  # PUT
                        # For selling, try to get a better exit price (0.1% above market)
                        trade_price = round(latest_price * 1.001, 4)
                        logger.info(
                            f"AI price management: Setting PUT price to {trade_price} (market: {latest_price})",
                        )
            except Exception as e:
                logger.error(f"Error in AI price management: {e}")
                # Fall back to market price if there's an error
                trade_price = None

        # Log strategy details
        strategy_details = {
            "direction": strategy.get("direction", "unknown"),
            "confidence": strategy.get("confidence", 0),
            "duration": duration,
            "technical_indicators": strategy.get("technical_indicators", {}),
            "price": trade_price if trade_price else "market price",
        }

        self._notify_status(
            {
                "status": "trading",
                "message": f"Executing {contract_type} trade for {duration}{duration_unit} on {symbol}",
                "amount": trade_amount,
                "price": trade_price if trade_price else "market price",
                "strategy": strategy_details,
            },
        )

        try:
            response = await self.client.buy_contract(
                symbol=symbol,
                contract_type=contract_type,
                amount=trade_amount,
                duration=duration,
                duration_unit=duration_unit,
                price=trade_price,
            )

            if "error" in response:
                error_msg = response.get("error", {}).get("message", "Unknown error")
                logger.error(f"Trade execution failed: {error_msg}")
                self._notify_status({"status": "error", "message": f"Trade failed: {error_msg}"})
                return None

            contract_id = response.get("buy", {}).get("contract_id")

            if not contract_id:
                logger.error("No contract ID in response")
                self._notify_status({"status": "error", "message": "No contract ID in response"})
                return None

            self._notify_status(
                {
                    "status": "executed",
                    "message": f"Trade executed: Contract ID {contract_id}",
                    "contract_id": contract_id,
                    "amount": trade_amount,
                    "strategy": strategy_details,
                },
            )

            return contract_id

        except Exception as e:
            error_msg = f"Exception during trade execution: {str(e)}"
            logger.error(error_msg)
            self._notify_status({"status": "error", "message": error_msg})
            return None

    async def monitor_trade(self, contract_id):
        """Monitor an active trade until it completes."""
        if not contract_id:
            return None

        self._notify_status(
            {
                "status": "monitoring",
                "message": f"Monitoring trade {contract_id}...",
                "contract_id": contract_id,
            },
        )

        # Check trade status every 5 seconds until it's closed
        retry_count = 0
        max_retries = 3
        update_count = 0
        _start_time = datetime.datetime.now()  # noqa: F841 - kept for potential future use
        entry_price = None
        contract_type = None
        duration = None
        symbol = None

        while True:
            try:
                contract_info = await self.client.check_trade_status(contract_id)

                # Reset retry counter on successful API call
                retry_count = 0

                if not contract_info:
                    await asyncio.sleep(5)
                    continue

                status = contract_info.get("status")

                # Store initial contract details on first successful check
                if entry_price is None:
                    entry_price = contract_info.get("buy_price", 0)
                    contract_type = contract_info.get("contract_type", "")
                    duration = contract_info.get("date_expiry", 0) - contract_info.get(
                        "date_start", 0,
                    )
                    symbol = contract_info.get("underlying", "")

                # Get current price and profit/loss
                current_spot = contract_info.get("current_spot", 0)
                _current_spot_time = contract_info.get("current_spot_time", 0)  # noqa: F841
                entry_spot = contract_info.get("entry_spot", 0)

                # Calculate profit/loss percentage
                if entry_spot and current_spot:
                    if contract_type == "CALL":
                        pnl_percent = (current_spot - entry_spot) / entry_spot * 100
                    else:  # PUT
                        pnl_percent = (entry_spot - current_spot) / entry_spot * 100
                else:
                    pnl_percent = 0

                # Calculate remaining time
                if duration:
                    expiry_time = contract_info.get("date_expiry", 0)
                    current_time = int(datetime.datetime.now().timestamp())
                    remaining_seconds = max(0, expiry_time - current_time)
                    remaining_time = str(datetime.timedelta(seconds=remaining_seconds)).split(".")[
                        0
                    ]
                else:
                    remaining_time = "Unknown"

                # Send periodic updates (every 5 seconds)
                update_count += 1
                if update_count % 1 == 0:  # Every check
                    self._notify_status(
                        {
                            "status": "update",
                            "message": f"Trade in progress - {remaining_time} remaining",
                            "contract_id": contract_id,
                            "current_price": current_spot,
                            "entry_price": entry_spot,
                            "pnl_percent": pnl_percent,
                            "remaining_time": remaining_time,
                            "contract_type": contract_type,
                            "symbol": symbol,
                        },
                    )

                if status in ["sold", "expired"]:
                    # Trade is complete
                    buy_price = contract_info.get("buy_price", 0)
                    sell_price = contract_info.get("sell_price", 0)
                    profit = sell_price - buy_price

                    # Update trade count
                    self.trade_count += 1
                    if profit > 0:
                        self.successful_trades += 1

                    # Create trade result
                    trade_result = {
                        "contract_id": contract_id,
                        "symbol": self.current_symbol,
                        "profit": profit,
                        "status": status,
                        "contract_info": contract_info,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }

                    # Update performance metrics
                    self.update_performance_metrics(trade_result)

                    # Adjust trade size based on performance
                    if self.trade_count % 5 == 0:  # Every 5 trades
                        self.adjust_trade_size()

                    result_message = (
                        f"Trade completed: {'Profit' if profit >= 0 else 'Loss'} ${abs(profit):.2f}"
                    )
                    logger.info(result_message)

                    self._notify_status(
                        {
                            "status": "completed",
                            "message": result_message,
                            "profit": profit,
                            "contract_info": contract_info,
                            "performance": {
                                "win_rate": self.successful_trades / self.trade_count
                                if self.trade_count > 0
                                else 0,
                                "net_profit": self.daily_profit - self.daily_loss,
                            },
                        },
                    )

                    return trade_result

                # If trade is still active, wait before checking again
                await asyncio.sleep(5)

            except Exception as e:
                # Log the error
                logger.error(f"Error checking trade status: {e}")

                # Notify UI of the error
                self._notify_status(
                    {"status": "error", "message": f"Error checking trade status: {str(e)}"},
                )

                # Increment retry counter
                retry_count += 1

                # If we've reached max retries, log a warning and continue
                if retry_count >= max_retries:
                    logger.warning(
                        f"Reached maximum retries ({max_retries}) for checking trade status. Continuing monitoring.",
                    )
                    retry_count = 0  # Reset counter to avoid continuous warnings

                # Wait before retrying
                await asyncio.sleep(5)

    async def trading_cycle(
        self, symbol, granularity, manual_price=None, use_ai_price=False, ai_managed_trading=False,
    ):
        """
        Run a complete trading cycle: generate strategy, execute trade, monitor, and update.

        Args:
            symbol: Market symbol to trade
            granularity: Timeframe granularity in seconds
            manual_price: Optional manual price setting
            use_ai_price: Whether to use AI for price management
            ai_managed_trading: Whether to let AI fully manage trading strategy
        """
        # Store current trading parameters
        self.current_symbol = symbol
        self.current_granularity = granularity

        # Check trading conditions
        if not self.check_trading_conditions():
            if self.trading_paused:
                self._notify_status({"status": "paused", "message": self.pause_reason})
                return False

            # If not paused but conditions failed, it might be time for model update
            if self.last_model_training:
                hours_since_training = (
                    datetime.datetime.now() - self.last_model_training
                ).total_seconds() / 3600
                if hours_since_training > MODEL_UPDATE_FREQUENCY:
                    logger.info(f"Retraining model after {hours_since_training:.1f} hours...")
                    success = await self.train_model(symbol, granularity)
                    if not success:
                        return False

        # Check if model needs initial training
        if not self.last_model_training:
            logger.info("No trained model found. Training new model...")
            success = await self.train_model(symbol, granularity)
            if not success:
                return False

        # Generate trading strategy
        strategy = await self.generate_strategy(symbol, granularity)

        if not strategy:
            return False

        # Check confidence threshold (use custom threshold if provided)
        threshold_to_use = (
            self.custom_confidence_threshold
            if self.custom_confidence_threshold is not None
            else CONFIDENCE_THRESHOLD
        )
        if strategy.get("confidence", 0) < threshold_to_use:
            logger.info(
                f"Strategy confidence too low: {strategy.get('confidence', 0):.2f} < {threshold_to_use}",
            )
            self._notify_status(
                {
                    "status": "skipped",
                    "message": f"Strategy confidence too low: {strategy.get('confidence', 0):.2f}",
                },
            )
            return False

        # Determine trade amount and parameters
        trade_amount = self.current_trade_amount

        # If AI-managed trading is enabled, adjust trade parameters
        if ai_managed_trading:
            # Get market volatility and recent performance
            try:
                # Analyze recent candles for volatility
                recent_candles = await self.client.get_candles(symbol, granularity, count=20)
                if recent_candles and len(recent_candles) > 10:
                    # Calculate volatility as normalized price range
                    highs = [float(candle["high"]) for candle in recent_candles]
                    lows = [float(candle["low"]) for candle in recent_candles]
                    closes = [float(candle["close"]) for candle in recent_candles]

                    avg_price = sum(closes) / len(closes)
                    price_ranges = [(h - low) / avg_price for h, low in zip(highs, lows)]
                    volatility = sum(price_ranges) / len(price_ranges)

                    # Adjust trade amount based on volatility and win rate
                    win_rate = (
                        self.successful_trades / self.trade_count if self.trade_count > 0 else 0.5
                    )

                    # Higher volatility = smaller position size
                    volatility_factor = max(0.5, min(1.5, 1.0 / (volatility * 10)))

                    # Higher win rate = larger position size
                    win_rate_factor = max(0.5, min(1.5, win_rate * 2))

                    # Calculate optimal trade amount
                    optimal_amount = self.base_trade_amount * volatility_factor * win_rate_factor

                    # Apply limits and round to 2 decimal places
                    trade_amount = round(
                        max(5.0, min(optimal_amount, self.base_trade_amount * 2)), 2,
                    )

                    logger.info(
                        f"AI-managed trading: Adjusted trade amount to ${trade_amount} "
                        + f"(volatility: {volatility:.4f}, win rate: {win_rate:.2f})",
                    )

                    # Also adjust strategy parameters based on market conditions
                    if strategy:
                        # Adjust duration based on volatility
                        if volatility > 0.02:  # Very high volatility
                            strategy["duration"] = max(
                                30, int(strategy["duration"] * 0.7),
                            )  # Shorter duration
                        elif volatility < 0.005:  # Low volatility
                            strategy["duration"] = min(
                                300, int(strategy["duration"] * 1.3),
                            )  # Longer duration

                        logger.info(
                            f"AI-managed trading: Adjusted trade duration to {strategy['duration']}s",
                        )
            except Exception as e:
                logger.error(f"Error in AI-managed trading adjustment: {e}")

        # Execute trade with determined parameters
        contract_id = await self.execute_trade(
            symbol=symbol,
            strategy=strategy,
            amount=trade_amount,
            price=manual_price,
            use_ai_price=use_ai_price
            or ai_managed_trading,  # Enable AI price if either option is on
        )

        if not contract_id:
            return False

        # Monitor trade until completion
        trade_result = await self.monitor_trade(contract_id)

        if not trade_result:
            return False

        # Update strategy based on results
        updated_strategy = self.trading_ai.update_strategy_from_results([trade_result])

        # If strategy is None, we need to recalculate
        if updated_strategy is None:
            logger.info("Recalculating strategy after trade...")
            await self.generate_strategy(symbol, granularity, recalculate=True)

        # Generate performance report every 10 trades
        if self.trade_count % 10 == 0:
            report = self.get_performance_report()
            logger.info(
                f"Performance report after {self.trade_count} trades: "
                f"Win rate: {report['summary']['win_rate']:.2f}, "
                f"Net profit: ${report['summary']['net_profit']:.2f}",
            )

            # Notify UI of performance report
            self._notify_status(
                {
                    "status": "report",
                    "message": f"Performance report after {self.trade_count} trades",
                    "report": report,
                },
            )

        return True

    async def run_trading_loop(
        self,
        symbol,
        granularity,
        manual_price=None,
        use_ai_price=False,
        ai_managed_trading=False,
        confidence_threshold=None,
    ):
        """
        Run the trading loop continuously.

        Args:
            symbol: Market symbol to trade
            granularity: Timeframe granularity in seconds
            manual_price: Optional manual price setting
            use_ai_price: Whether to use AI for price management
            ai_managed_trading: Whether to let AI fully manage trading strategy
            confidence_threshold: Custom confidence threshold (overrides settings.CONFIDENCE_THRESHOLD)
        """
        self.is_running = True
        self.trading_paused = False
        self.pause_reason = None

        # Store trading settings
        self.manual_price = manual_price
        self.use_ai_price = use_ai_price
        self.ai_managed_trading = ai_managed_trading
        self.custom_confidence_threshold = confidence_threshold

        # Reset counters for new session
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.session_start_time = datetime.datetime.now()
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.max_drawdown = 0.0

        # Reset trade size to base amount
        self.current_trade_amount = self.base_trade_amount
        self.trade_size_multiplier = 1.0

        # Prepare trading settings message
        price_setting = "market price"
        if manual_price:
            price_setting = f"manual price: {manual_price}"
        elif use_ai_price:
            price_setting = "AI-managed price"

        trading_mode = "standard"
        if ai_managed_trading:
            trading_mode = "AI-managed trading"

        self._notify_status(
            {
                "status": "started",
                "message": f"Auto-trading started for {symbol} with {granularity}s granularity",
                "settings": {
                    "trade_amount": self.current_trade_amount,
                    "max_daily_loss": MAX_DAILY_LOSS,
                    "confidence_threshold": self.custom_confidence_threshold
                    if self.custom_confidence_threshold is not None
                    else CONFIDENCE_THRESHOLD,
                    "price_setting": price_setting,
                    "trading_mode": trading_mode,
                },
            },
        )

        # Reset day change time
        day_change_time = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) + datetime.timedelta(days=1)

        # Initialize performance tracking for this market
        if symbol not in self.performance_by_market:
            self.performance_by_market[symbol] = {"trades": 0, "wins": 0, "profit": 0, "loss": 0}

        while self.is_running:
            try:
                # Check if we need to reset daily counters
                now = datetime.datetime.now()
                if now >= day_change_time:
                    logger.info("New day started. Resetting daily counters.")

                    # Log daily summary
                    daily_summary = {
                        "date": day_change_time.strftime("%Y-%m-%d"),
                        "trades": self.trade_count,
                        "wins": self.successful_trades,
                        "win_rate": self.successful_trades / self.trade_count
                        if self.trade_count > 0
                        else 0,
                        "profit": self.daily_profit,
                        "loss": self.daily_loss,
                        "net": self.daily_profit - self.daily_loss,
                    }

                    # Log to file
                    log_file = os.path.join(self.logs_dir, "daily_summary.json")
                    try:
                        with open(log_file, "a") as f:
                            f.write(json.dumps(daily_summary) + "\n")
                    except Exception as e:
                        logger.error(f"Error logging daily summary: {e}")

                    # Reset counters
                    self.daily_loss = 0.0
                    self.daily_profit = 0.0

                    # Reset trading pause if it was due to daily loss
                    if self.trading_paused and "daily loss limit" in self.pause_reason.lower():
                        self.trading_paused = False
                        self.pause_reason = None
                        logger.info("Trading resumed after daily reset")

                    # Update day change time
                    day_change_time = now.replace(
                        hour=0, minute=0, second=0, microsecond=0,
                    ) + datetime.timedelta(days=1)

                # Check if trading is paused
                if self.trading_paused:
                    logger.info(f"Trading is paused: {self.pause_reason}")
                    self._notify_status({"status": "paused", "message": self.pause_reason})
                    await asyncio.sleep(60)  # Check again in a minute
                    continue

                # Run a trading cycle
                cycle_success = await self.trading_cycle(
                    symbol=symbol,
                    granularity=granularity,
                    manual_price=self.manual_price,
                    use_ai_price=self.use_ai_price,
                    ai_managed_trading=self.ai_managed_trading,
                )

                # Determine wait time based on cycle success and market conditions
                if cycle_success:
                    # Successful cycle, wait standard time
                    wait_time = 30
                elif self.trading_paused:
                    # Trading paused, wait longer
                    wait_time = 60
                else:
                    # Unsuccessful cycle but not paused, wait medium time
                    wait_time = 45

                # Wait before the next cycle
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self._notify_status({"status": "error", "message": f"Trading error: {str(e)}"})
                await asyncio.sleep(60)  # Wait longer after an error

        # Generate final performance report
        final_report = self.get_performance_report()

        self._notify_status(
            {"status": "stopped", "message": "Auto-trading stopped", "final_report": final_report},
        )

        logger.info(
            f"Trading session ended. Final report: Win rate: {final_report['summary']['win_rate']:.2f}, "
            f"Net profit: ${final_report['summary']['net_profit']:.2f}",
        )

    def start_trading(
        self,
        symbol,
        granularity,
        manual_price=None,
        use_ai_price=False,
        ai_managed_trading=False,
        confidence_threshold=None,
    ):
        """
        Start the auto-trading process in a separate thread.

        Args:
            symbol: Market symbol to trade
            granularity: Timeframe granularity in seconds
            manual_price: Optional manual price setting
            use_ai_price: Whether to use AI for price management
            ai_managed_trading: Whether to let AI fully manage trading strategy
            confidence_threshold: Custom confidence threshold (overrides settings.CONFIDENCE_THRESHOLD)
        """
        # If AI-managed trading is enabled, also enable AI price management
        if ai_managed_trading:
            use_ai_price = True
        if self.is_running:
            logger.warning("Trading is already running")
            return False

        # Create a new event loop for the trading thread
        self.loop = asyncio.new_event_loop()

        # Start the trading loop in a separate thread
        self.trading_thread = threading.Thread(
            target=self._run_trading_thread,
            args=(
                self.loop,
                symbol,
                granularity,
                manual_price,
                use_ai_price,
                ai_managed_trading,
                confidence_threshold,
            ),
            daemon=True,
        )
        self.trading_thread.start()

        return True

    def _run_trading_thread(
        self,
        loop,
        symbol,
        granularity,
        manual_price=None,
        use_ai_price=False,
        ai_managed_trading=False,
        confidence_threshold=None,
    ):
        """Run the trading loop in a separate thread."""
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.run_trading_loop(
                symbol,
                granularity,
                manual_price,
                use_ai_price,
                ai_managed_trading,
                confidence_threshold,
            ),
        )

    def stop_trading(self):
        """Stop the auto-trading process."""
        if not self.is_running:
            logger.warning("Trading is not running")
            return False

        logger.info("Stopping auto-trading...")
        self.is_running = False

        # Wait for the trading thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)

        return True

    def get_status(self):
        """Get the current status of the auto-trader."""
        win_rate = self.successful_trades / self.trade_count if self.trade_count > 0 else 0

        return {
            "is_running": self.is_running,
            "trading_paused": self.trading_paused,
            "pause_reason": self.pause_reason,
            "symbol": self.current_symbol,
            "granularity": self.current_granularity,
            "trade_count": self.trade_count,
            "successful_trades": self.successful_trades,
            "win_rate": win_rate,
            "daily_loss": self.daily_loss,
            "daily_profit": self.daily_profit,
            "net_profit": self.daily_profit - self.daily_loss,
            "max_drawdown": self.max_drawdown,
            "current_trade_amount": self.current_trade_amount,
            "last_model_training": self.last_model_training.isoformat()
            if self.last_model_training
            else None,
            "model_metrics": self.trading_ai.performance_metrics if self.trading_ai else None,
            "session_duration": str(datetime.datetime.now() - self.session_start_time).split(".")[
                0
            ],
        }

    def update_performance_metrics(self, trade_result):
        """Update performance metrics based on a completed trade."""
        if not trade_result:
            return

        # Extract trade details
        contract_info = trade_result.get("contract_info", {})
        profit = trade_result.get("profit", 0)
        symbol = trade_result.get("symbol", self.current_symbol)
        timestamp = datetime.datetime.now()

        # Update trade history
        trade_record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "profit": profit,
            "duration": contract_info.get("duration", 0),
            "direction": contract_info.get("contract_type", ""),
            "amount": contract_info.get("buy_price", 0),
        }
        self.trade_history.append(trade_record)

        # Update hourly performance
        hour = timestamp.hour
        self.performance_by_hour[hour]["trades"] += 1
        if profit > 0:
            self.performance_by_hour[hour]["wins"] += 1

        # Update market performance
        if symbol not in self.performance_by_market:
            self.performance_by_market[symbol] = {"trades": 0, "wins": 0, "profit": 0, "loss": 0}

        self.performance_by_market[symbol]["trades"] += 1
        if profit > 0:
            self.performance_by_market[symbol]["wins"] += 1
            self.performance_by_market[symbol]["profit"] += profit
        else:
            self.performance_by_market[symbol]["loss"] += abs(profit)

        # Update balance and drawdown
        if profit > 0:
            self.daily_profit += profit
            self.current_balance += profit
        else:
            self.daily_loss += abs(profit)
            self.current_balance -= abs(profit)

        # Update peak balance and max drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        current_drawdown = self.peak_balance - self.current_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Log trade result
        self.log_trade(trade_record)

    def log_trade(self, trade_record):
        """Log trade details to a file."""
        log_file = os.path.join(
            self.logs_dir, f"trades_{datetime.datetime.now().strftime('%Y%m%d')}.json",
        )

        try:
            # Convert timestamp to string for JSON serialization
            log_entry = trade_record.copy()
            log_entry["timestamp"] = log_entry["timestamp"].isoformat()

            # Append to log file
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    def adjust_trade_size(self):
        """Adjust trade size based on performance and risk management."""
        # Get win rate
        win_rate = self.successful_trades / self.trade_count if self.trade_count > 0 else 0.5

        # Calculate risk factor based on drawdown
        risk_factor = 1.0
        if self.peak_balance > 0:
            drawdown_percentage = self.max_drawdown / self.peak_balance
            if drawdown_percentage > 0.2:  # High drawdown
                risk_factor = 0.5  # Reduce risk
            elif drawdown_percentage > 0.1:  # Moderate drawdown
                risk_factor = 0.75  # Slightly reduce risk

        # Adjust based on win rate
        if win_rate > 0.6:  # Good performance
            win_factor = 1.2
        elif win_rate < 0.4:  # Poor performance
            win_factor = 0.8
        else:  # Average performance
            win_factor = 1.0

        # Calculate new multiplier
        new_multiplier = risk_factor * win_factor

        # Smooth the transition
        self.trade_size_multiplier = 0.7 * self.trade_size_multiplier + 0.3 * new_multiplier

        # Apply limits
        self.trade_size_multiplier = max(0.5, min(1.5, self.trade_size_multiplier))

        # Update trade amount
        self.current_trade_amount = round(self.base_trade_amount * self.trade_size_multiplier, 2)

        logger.info(
            f"Trade size adjusted: ${self.current_trade_amount} (multiplier: {self.trade_size_multiplier:.2f})",
        )

    def check_trading_conditions(self):
        """Check if trading should continue or be paused based on performance and risk metrics."""
        # Check daily loss limit
        if self.daily_loss >= MAX_DAILY_LOSS:
            self.trading_paused = True
            self.pause_reason = f"Daily loss limit reached: ${self.daily_loss:.2f}"
            logger.warning(self.pause_reason)
            return False

        # Check drawdown
        if self.peak_balance > 0:
            drawdown_percentage = self.max_drawdown / self.peak_balance
            if drawdown_percentage > 0.3:  # 30% drawdown
                self.trading_paused = True
                self.pause_reason = f"Maximum drawdown threshold reached: {drawdown_percentage:.1%}"
                logger.warning(self.pause_reason)
                return False

        # Check win rate if we have enough trades
        if self.trade_count >= 10:
            win_rate = self.successful_trades / self.trade_count
            if win_rate < 0.3:  # Very poor performance
                self.trading_paused = True
                self.pause_reason = f"Win rate too low: {win_rate:.1%}"
                logger.warning(self.pause_reason)
                return False

        # Check if model needs retraining
        if self.last_model_training:
            hours_since_training = (
                datetime.datetime.now() - self.last_model_training
            ).total_seconds() / 3600
            if hours_since_training > MODEL_UPDATE_FREQUENCY:
                logger.info(
                    f"Model update needed: {hours_since_training:.1f} hours since last training",
                )
                return False

        # All conditions passed
        self.trading_paused = False
        self.pause_reason = None
        return True

    def get_performance_report(self):
        """Generate a comprehensive performance report."""
        win_rate = self.successful_trades / self.trade_count if self.trade_count > 0 else 0

        # Calculate hourly performance
        best_hour = max(
            self.performance_by_hour.items(),
            key=lambda x: x[1]["wins"] / x[1]["trades"] if x[1]["trades"] > 0 else 0,
        )
        worst_hour = min(
            self.performance_by_hour.items(),
            key=lambda x: x[1]["wins"] / x[1]["trades"] if x[1]["trades"] > 0 else 1,
        )

        # Calculate market performance
        best_market = None
        worst_market = None
        if self.performance_by_market:
            best_market = max(
                self.performance_by_market.items(),
                key=lambda x: x[1]["wins"] / x[1]["trades"] if x[1]["trades"] > 0 else 0,
            )
            worst_market = min(
                self.performance_by_market.items(),
                key=lambda x: x[1]["wins"] / x[1]["trades"] if x[1]["trades"] > 0 else 1,
            )

        report = {
            "summary": {
                "total_trades": self.trade_count,
                "win_rate": win_rate,
                "net_profit": self.daily_profit - self.daily_loss,
                "max_drawdown": self.max_drawdown,
                "session_duration": str(datetime.datetime.now() - self.session_start_time).split(
                    ".",
                )[0],
            },
            "hourly_performance": {
                "best_hour": {
                    "hour": best_hour[0],
                    "win_rate": best_hour[1]["wins"] / best_hour[1]["trades"]
                    if best_hour[1]["trades"] > 0
                    else 0,
                    "trades": best_hour[1]["trades"],
                },
                "worst_hour": {
                    "hour": worst_hour[0],
                    "win_rate": worst_hour[1]["wins"] / worst_hour[1]["trades"]
                    if worst_hour[1]["trades"] > 0
                    else 0,
                    "trades": worst_hour[1]["trades"],
                },
            },
            "market_performance": {},
        }

        if best_market:
            report["market_performance"]["best_market"] = {
                "symbol": best_market[0],
                "win_rate": best_market[1]["wins"] / best_market[1]["trades"]
                if best_market[1]["trades"] > 0
                else 0,
                "trades": best_market[1]["trades"],
                "net_profit": best_market[1]["profit"] - best_market[1]["loss"],
            }

        if worst_market:
            report["market_performance"]["worst_market"] = {
                "symbol": worst_market[0],
                "win_rate": worst_market[1]["wins"] / worst_market[1]["trades"]
                if worst_market[1]["trades"] > 0
                else 0,
                "trades": worst_market[1]["trades"],
                "net_profit": worst_market[1]["profit"] - worst_market[1]["loss"],
            }

        return report

import asyncio
import datetime
import logging
import threading
import time
from typing import Dict, List, Optional

from api.deriv_client import DerivClient
from ml.trading_ai import TradingAI
from config.settings import (
    DERIV_API_TOKEN, TRADE_AMOUNT, TRADE_CURRENCY, 
    MAX_DAILY_LOSS, MAX_CONSECUTIVE_LOSSES, CONFIDENCE_THRESHOLD,
    TRAINING_DAYS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('auto_trader')

class AutoTrader:
    """
    Automated trading manager that coordinates between the TradingAI and DerivClient
    to execute trades based on generated strategies.
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
        self.daily_loss = 0.0
        self.trade_count = 0
        self.successful_trades = 0
        self.last_model_training = None
        self.status_callbacks = []
        
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
                
    async def train_model(self, symbol, granularity):
        """Train the AI model on historical data for the specified symbol and granularity."""
        self._notify_status({"status": "training", "message": "Collecting historical data..."})
        
        # Get historical data for training
        historical_data = await self.client.get_historical_data(symbol, granularity, days=TRAINING_DAYS)
        
        if not historical_data or len(historical_data) < 100:
            message = f"Insufficient historical data: {len(historical_data) if historical_data else 0} candles"
            logger.warning(message)
            self._notify_status({"status": "error", "message": message})
            return False
            
        self._notify_status({
            "status": "training", 
            "message": f"Training model on {len(historical_data)} candles..."
        })
        
        # Train the model
        metrics = self.trading_ai.train_model(historical_data)
        
        if not metrics:
            message = "Model training failed"
            logger.error(message)
            self._notify_status({"status": "error", "message": message})
            return False
            
        self.last_model_training = datetime.datetime.now()
        
        self._notify_status({
            "status": "trained", 
            "message": "Model training completed",
            "metrics": metrics
        })
        
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
            
        # Generate strategy
        strategy = self.trading_ai.generate_strategy(recent_data, recalculate=recalculate)
        
        if not strategy:
            message = "Failed to generate strategy"
            logger.warning(message)
            self._notify_status({"status": "error", "message": message})
            return None
            
        self._notify_status({
            "status": "strategy", 
            "message": f"Strategy generated: {strategy['direction']} with {strategy['confidence']:.2f} confidence",
            "strategy": strategy
        })
        
        return strategy
        
    async def execute_trade(self, symbol, strategy):
        """Execute a trade based on the generated strategy."""
        if not strategy or strategy['confidence'] < CONFIDENCE_THRESHOLD:
            message = f"Strategy confidence too low: {strategy['confidence'] if strategy else 0:.2f}"
            logger.info(message)
            self._notify_status({"status": "skipped", "message": message})
            return None
            
        # Check if we've exceeded the daily loss limit
        if self.daily_loss >= MAX_DAILY_LOSS:
            message = f"Daily loss limit reached: ${self.daily_loss:.2f}"
            logger.warning(message)
            self._notify_status({"status": "stopped", "message": message})
            return None
            
        # Execute the trade
        contract_type = strategy['contract_type']
        duration = strategy['duration']
        duration_unit = strategy['duration_unit']
        
        self._notify_status({
            "status": "trading", 
            "message": f"Executing {contract_type} trade for {duration}{duration_unit} on {symbol}"
        })
        
        response = await self.client.buy_contract(
            symbol=symbol,
            contract_type=contract_type,
            amount=TRADE_AMOUNT,
            duration=duration,
            duration_unit=duration_unit
        )
        
        if not response or "error" in response:
            error_msg = response.get("error", {}).get("message", "Unknown error") if response else "Failed to execute trade"
            logger.error(f"Trade execution failed: {error_msg}")
            self._notify_status({"status": "error", "message": f"Trade failed: {error_msg}"})
            return None
            
        contract_id = response.get("buy", {}).get("contract_id")
        
        if not contract_id:
            logger.error("No contract ID in response")
            self._notify_status({"status": "error", "message": "No contract ID in response"})
            return None
            
        self._notify_status({
            "status": "executed", 
            "message": f"Trade executed: Contract ID {contract_id}",
            "contract_id": contract_id
        })
        
        self.trade_count += 1
        
        return contract_id
        
    async def monitor_trade(self, contract_id):
        """Monitor an active trade until it completes."""
        if not contract_id:
            return None
            
        self._notify_status({
            "status": "monitoring", 
            "message": f"Monitoring trade {contract_id}..."
        })
        
        # Check trade status every 5 seconds until it's closed
        while True:
            contract_info = await self.client.check_trade_status(contract_id)
            
            if not contract_info:
                await asyncio.sleep(5)
                continue
                
            status = contract_info.get("status")
            
            if status in ["sold", "expired"]:
                # Trade is complete
                buy_price = contract_info.get("buy_price", 0)
                sell_price = contract_info.get("sell_price", 0)
                profit = sell_price - buy_price
                
                # Update daily loss counter
                if profit < 0:
                    self.daily_loss += abs(profit)
                else:
                    self.successful_trades += 1
                    
                result_message = f"Trade completed: {'Profit' if profit >= 0 else 'Loss'} ${abs(profit):.2f}"
                logger.info(result_message)
                
                self._notify_status({
                    "status": "completed", 
                    "message": result_message,
                    "profit": profit,
                    "contract_info": contract_info
                })
                
                return {
                    "contract_id": contract_id,
                    "profit": profit,
                    "status": status,
                    "contract_info": contract_info
                }
                
            # Wait before checking again
            await asyncio.sleep(5)
            
    async def trading_cycle(self, symbol, granularity):
        """Run a complete trading cycle: generate strategy, execute trade, monitor, and update."""
        # Store current trading parameters
        self.current_symbol = symbol
        self.current_granularity = granularity
        
        # Check if model needs training
        if not self.last_model_training:
            logger.info("No trained model found. Training new model...")
            success = await self.train_model(symbol, granularity)
            if not success:
                return False
                
        # Generate trading strategy
        strategy = await self.generate_strategy(symbol, granularity)
        
        if not strategy:
            return False
            
        # Execute trade
        contract_id = await self.execute_trade(symbol, strategy)
        
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
            
        return True
        
    async def run_trading_loop(self, symbol, granularity):
        """Run the trading loop continuously."""
        self.is_running = True
        self.daily_loss = 0.0  # Reset daily loss counter
        
        self._notify_status({
            "status": "started", 
            "message": f"Auto-trading started for {symbol} with {granularity}s granularity"
        })
        
        # Reset day change time
        day_change_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        
        while self.is_running:
            try:
                # Check if we need to reset daily counters
                now = datetime.datetime.now()
                if now >= day_change_time:
                    logger.info("New day started. Resetting daily counters.")
                    self.daily_loss = 0.0
                    day_change_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
                    
                # Run a trading cycle
                await self.trading_cycle(symbol, granularity)
                
                # Wait a bit before the next cycle to avoid too frequent trading
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self._notify_status({"status": "error", "message": f"Trading error: {str(e)}"})
                await asyncio.sleep(60)  # Wait longer after an error
                
        self._notify_status({
            "status": "stopped", 
            "message": "Auto-trading stopped"
        })
        
    def start_trading(self, symbol, granularity):
        """Start the auto-trading process in a separate thread."""
        if self.is_running:
            logger.warning("Trading is already running")
            return False
            
        # Create a new event loop for the trading thread
        self.loop = asyncio.new_event_loop()
        
        # Start the trading loop in a separate thread
        self.trading_thread = threading.Thread(
            target=self._run_trading_thread,
            args=(self.loop, symbol, granularity),
            daemon=True
        )
        self.trading_thread.start()
        
        return True
        
    def _run_trading_thread(self, loop, symbol, granularity):
        """Run the trading loop in a separate thread."""
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_trading_loop(symbol, granularity))
        
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
        return {
            "is_running": self.is_running,
            "symbol": self.current_symbol,
            "granularity": self.current_granularity,
            "trade_count": self.trade_count,
            "successful_trades": self.successful_trades,
            "daily_loss": self.daily_loss,
            "last_model_training": self.last_model_training.isoformat() if self.last_model_training else None,
            "model_metrics": self.trading_ai.performance_metrics if self.trading_ai else None
        }
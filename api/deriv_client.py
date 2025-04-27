import asyncio
import datetime
from deriv_api import DerivAPI

class DerivClient:
    """Client for interacting with the Deriv API."""

    def __init__(self, app_id='70789', api_token=None):
        self.app_id = app_id
        self.api_token = api_token
        self.active_trades = {}  # Dictionary to track active trades
        self.trade_history = []  # List to track trade history
        self.consecutive_losses = 0  # Counter for consecutive losses

    async def _get_api_instance(self):
        """Get an authenticated API instance."""
        if self.api_token:
            return DerivAPI(app_id=self.app_id, token=self.api_token)
        else:
            return DerivAPI(app_id=self.app_id)

    async def get_active_symbols(self):
        """Fetch and return active market symbols."""
        try:
            api = await self._get_api_instance()
            response = await api.active_symbols({"active_symbols": "brief", "product_type": "basic"})
            active_symbols = response.get('active_symbols', [])
            symbols_dict = {}
            for symbol in active_symbols:
                # Check if the market is closed
                is_closed = not symbol.get('exchange_is_open', True)
                market_name = f"{symbol['market_display_name']} - {symbol['display_name']}"

                # Add closed indicator to market name if closed
                if is_closed:
                    market_name = f"{market_name} [CLOSED]"

                symbols_dict[market_name] = symbol['symbol']
            return symbols_dict
        except Exception as e:
            print(f"Error fetching active symbols: {e}")
            return {}

    async def get_candles(self, symbol, granularity, count=50):
        """Retrieve candle data for a specific symbol and granularity."""
        try:
            api = await self._get_api_instance()
            response = await api.ticks_history({
                "ticks_history": symbol,
                "count": count,
                "end": "latest",
                "style": "candles",
                "granularity": granularity
            })
            return response.get('candles', [])
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return []

    async def get_historical_data(self, symbol, granularity, days=7):
        """Retrieve historical candle data for a specific symbol and granularity for the specified number of days."""
        try:
            api = await self._get_api_instance()
            # Calculate start time (days ago from now)
            end_time = int(datetime.datetime.now().timestamp())
            start_time = end_time - (days * 24 * 60 * 60)

            response = await api.ticks_history({
                "ticks_history": symbol,
                "start": start_time,
                "end": end_time,
                "style": "candles",
                "granularity": granularity
            })
            return response.get('candles', [])
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return []

    async def buy_contract(self, symbol, contract_type, amount, duration, duration_unit="s"):
        """Buy a contract with the specified parameters."""
        if not self.api_token:
            print("API token is required for trading operations")
            return None

        try:
            api = await self._get_api_instance()

            # Prepare the buy request
            buy_params = {
                "buy": 1,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "symbol": symbol
                }
            }

            response = await api.buy(buy_params)

            if "error" in response:
                print(f"Error buying contract: {response['error']['message']}")
                return None

            # Store the trade in active trades
            contract_id = response.get("buy", {}).get("contract_id")
            if contract_id:
                self.active_trades[contract_id] = {
                    "symbol": symbol,
                    "contract_type": contract_type,
                    "amount": amount,
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "buy_response": response,
                    "status": "open",
                    "start_time": datetime.datetime.now()
                }

            return response
        except Exception as e:
            print(f"Error buying contract: {e}")
            return None

    async def check_trade_status(self, contract_id):
        """Check the status of a specific trade."""
        if not self.api_token or contract_id not in self.active_trades:
            return None

        try:
            api = await self._get_api_instance()

            response = await api.proposal_open_contract({
                "contract_id": contract_id,
                "subscribe": 0
            })

            if "error" in response:
                print(f"Error checking trade status: {response['error']['message']}")
                return None

            contract_info = response.get("proposal_open_contract", {})

            # Update trade information
            if contract_id in self.active_trades:
                self.active_trades[contract_id]["current_info"] = contract_info

                # Check if the contract is finished
                if contract_info.get("status") in ["sold", "expired"]:
                    self.active_trades[contract_id]["status"] = "closed"
                    self.active_trades[contract_id]["end_time"] = datetime.datetime.now()

                    # Calculate profit/loss
                    buy_price = contract_info.get("buy_price", 0)
                    sell_price = contract_info.get("sell_price", 0)
                    profit = sell_price - buy_price

                    self.active_trades[contract_id]["profit"] = profit

                    # Add to trade history
                    trade_record = self.active_trades[contract_id].copy()
                    self.trade_history.append(trade_record)

                    # Update consecutive losses counter
                    if profit < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

                    # Remove from active trades
                    del self.active_trades[contract_id]

            return contract_info
        except Exception as e:
            print(f"Error checking trade status: {e}")
            return None

    async def check_all_trades(self):
        """Check the status of all active trades."""
        results = {}
        for contract_id in list(self.active_trades.keys()):
            results[contract_id] = await self.check_trade_status(contract_id)
        return results

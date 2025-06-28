import logging
from datetime import UTC, datetime

from deriv_api import DerivAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deriv_client")


class DerivClient:
    """Client for interacting with the Deriv API."""

    def __init__(self, app_id="70789", api_token=None):
        self.app_id = app_id
        self.api_token = api_token
        self.active_trades = {}  # Dictionary to track active trades
        self.trade_history = []  # List to track trade history
        self.consecutive_losses = 0  # Counter for consecutive losses
        self.is_connected = False
        self.account_info = None

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

            # Explicitly authorize with the API token before fetching active symbols
            if self.api_token:
                auth_response = await api.authorize({"authorize": self.api_token})

                if "error" in auth_response:
                    error_msg = f"Authorization failed: {auth_response['error']['message']}"
                    logger.error(error_msg)
                    return {}

                # Update account info after successful authorization
                self.account_info = auth_response.get("authorize", {})

            response = await api.active_symbols(
                {"active_symbols": "brief", "product_type": "basic"},
            )
            active_symbols = response.get("active_symbols", [])
            symbols_dict = {}
            for symbol in active_symbols:
                # Check if the market is closed
                is_closed = not symbol.get("exchange_is_open", True)
                market_name = f"{symbol['market_display_name']} - {symbol['display_name']}"

                # Add closed indicator to market name if closed
                if is_closed:
                    market_name = f"{market_name} [CLOSED]"

                symbols_dict[market_name] = symbol["symbol"]
            return symbols_dict
        except Exception as e:
            print(f"Error fetching active symbols: {e}")
            return {}

    async def get_candles(self, symbol, granularity, count=50):
        """Retrieve candle data for a specific symbol and granularity."""
        try:
            api = await self._get_api_instance()

            # Explicitly authorize with the API token before fetching candles
            if self.api_token:
                auth_response = await api.authorize({"authorize": self.api_token})

                if "error" in auth_response:
                    error_msg = f"Authorization failed: {auth_response['error']['message']}"
                    logger.error(error_msg)
                    return []

                # Update account info after successful authorization
                self.account_info = auth_response.get("authorize", {})

            response = await api.ticks_history(
                {
                    "ticks_history": symbol,
                    "count": count,
                    "end": "latest",
                    "style": "candles",
                    "granularity": granularity,
                },
            )
            return response.get("candles", [])
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return []

    async def get_historical_data(self, symbol, granularity, days=7):
        """Retrieve historical candle data for a specific symbol and granularity for the specified number of days."""
        try:
            api = await self._get_api_instance()

            # Explicitly authorize with the API token before fetching historical data
            if self.api_token:
                auth_response = await api.authorize({"authorize": self.api_token})

                if "error" in auth_response:
                    error_msg = f"Authorization failed: {auth_response['error']['message']}"
                    logger.error(error_msg)
                    return []

                # Update account info after successful authorization
                self.account_info = auth_response.get("authorize", {})

            # Calculate count based on granularity and days
            # granularity is in seconds, so calculate how many candles in the given days
            seconds_per_day = 24 * 60 * 60
            total_seconds = days * seconds_per_day
            count = min(total_seconds // granularity, 5000)  # Cap at 5000 to avoid API limits

            # Use the same format as get_candles but with more data
            response = await api.ticks_history(
                {
                    "ticks_history": symbol,
                    "count": int(count),
                    "end": "latest",
                    "style": "candles",
                    "granularity": granularity,
                },
            )
            return response.get("candles", [])
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return []

    async def get_contract_proposal(
        self, symbol, contract_type, amount, duration, duration_unit="s",
    ):
        """
        Get a contract proposal to check if the trade is valid and get max stake limits.

        Args:
            symbol: Market symbol
            contract_type: Type of contract (CALL/PUT)
            amount: Trade amount
            duration: Contract duration
            duration_unit: Duration unit (s/m/h/d)

        Returns:
            Proposal information including max stake limits
        """
        if not self.api_token:
            error_msg = "API token is required for trading operations"
            print(error_msg)
            return {"error": {"message": error_msg}}

        try:
            api = await self._get_api_instance()

            # Explicitly authorize with the API token
            auth_response = await api.authorize({"authorize": self.api_token})

            if "error" in auth_response:
                error_msg = f"Authorization failed: {auth_response['error']['message']}"
                logger.error(error_msg)
                return {"error": {"message": error_msg}}

            # Update account info after successful authorization
            self.account_info = auth_response.get("authorize", {})

            # Prepare the proposal request
            proposal_params = {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": duration_unit,
                "symbol": symbol,
            }

            print(f"Requesting proposal with parameters: {proposal_params}")
            response = await api.proposal(proposal_params)

            if "error" in response:
                error_msg = f"Error getting proposal: {response['error']['message']}"
                print(error_msg)
                return response

            return response.get("proposal", {})
        except Exception as e:
            error_msg = f"Error getting contract proposal: {e}"
            print(error_msg)
            return {"error": {"message": error_msg}}

    async def buy_contract(
        self, symbol, contract_type, amount, duration, duration_unit="s", price=None,
    ):
        """
        Buy a contract with the specified parameters.

        Args:
            symbol: Market symbol
            contract_type: Type of contract (CALL/PUT)
            amount: Trade amount
            duration: Contract duration
            duration_unit: Duration unit (s/m/h/d)
            price: Contract purchase price (optional)

        Returns:
            API response
        """
        if not self.api_token:
            error_msg = "API token is required for trading operations"
            print(error_msg)
            return {"error": {"message": error_msg}}

        try:
            api = await self._get_api_instance()

            # Explicitly authorize with the API token before making a trade
            auth_response = await api.authorize({"authorize": self.api_token})

            if "error" in auth_response:
                error_msg = f"Authorization failed: {auth_response['error']['message']}"
                logger.error(error_msg)
                return {"error": {"message": error_msg}}

            # Update account info after successful authorization
            self.account_info = auth_response.get("authorize", {})

            # Check if the stake amount exceeds the maximum purchase price
            proposal = await self.get_contract_proposal(
                symbol, contract_type, amount, duration, duration_unit,
            )

            if "error" in proposal:
                # If there's an error with the proposal, check if it's related to the stake amount
                error_msg = proposal.get("error", {}).get("message", "")
                if "stake" in error_msg.lower() and "maximum" in error_msg.lower():
                    # Try to extract the maximum stake from the error message
                    import re

                    max_stake_match = re.search(r"maximum.*?(\d+(\.\d+)?)", error_msg)
                    if max_stake_match:
                        max_stake = float(max_stake_match.group(1))
                        print(f"Reducing stake amount from {amount} to {max_stake}")
                        # Use the maximum stake amount instead
                        amount = max_stake
                    else:
                        # If we can't extract the maximum, reduce by 25%
                        reduced_amount = amount * 0.75
                        print(f"Reducing stake amount from {amount} to {reduced_amount}")
                        amount = reduced_amount
                else:
                    # For other errors, return the error response
                    return proposal
            else:
                # Check if the proposal contains information about maximum stake
                max_stake = proposal.get("max_stake")
                if max_stake and float(amount) > float(max_stake):
                    print(f"Stake amount {amount} exceeds maximum {max_stake}, reducing to maximum")
                    amount = float(max_stake)

            # If price is not provided, get the current market price
            if price is None:
                print(f"No price provided for {symbol}, attempting to get current market price")
                # Get the latest tick to determine current price
                # Note: We don't use the "subscribe" parameter here as it causes validation errors
                tick_response = await api.ticks({"ticks": symbol})

                if "error" in tick_response:
                    error_msg = f"Error getting current price: {tick_response['error']['message']}"
                    print(error_msg)
                    return {"error": {"message": error_msg}}

                print(f"Tick response: {tick_response}")
                price = tick_response.get("tick", {}).get("quote")
                print(f"Retrieved price: {price}")

                if not price:
                    error_msg = "Could not determine current market price"
                    print(error_msg)
                    return {"error": {"message": error_msg}}

            # Ensure we have a valid price
            if price is None:
                error_msg = "Price is required for contract purchase"
                print(error_msg)
                return {"error": {"message": error_msg}}

            # Convert price to string to ensure it's properly formatted for the API
            price = str(price)

            # Prepare the buy request
            buy_params = {
                "buy": 1,
                "price": price,  # Include price at the top level
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "symbol": symbol,
                },
            }

            print(f"Using price: {price} for contract purchase")

            print(f"Sending buy request with parameters: {buy_params}")
            response = await api.buy(buy_params)

            if "error" in response:
                error_msg = f"Error buying contract: {response['error']['message']}"
                print(error_msg)
                return response

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
                    "start_time": datetime.now(tz=UTC),
                }

            return response
        except Exception as e:
            error_msg = f"Error buying contract: {e}"
            print(error_msg)
            return {"error": {"message": error_msg}}

    async def check_trade_status(self, contract_id):
        """Check the status of a specific trade."""
        if not self.api_token or contract_id not in self.active_trades:
            return None

        try:
            api = await self._get_api_instance()

            # Explicitly authorize with the API token before checking trade status
            auth_response = await api.authorize({"authorize": self.api_token})

            if "error" in auth_response:
                error_msg = f"Authorization failed: {auth_response['error']['message']}"
                logger.error(error_msg)
                return None

            # Update account info after successful authorization
            self.account_info = auth_response.get("authorize", {})

            response = await api.proposal_open_contract({"contract_id": contract_id})

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
                    self.active_trades[contract_id]["end_time"] = datetime.now(tz=UTC)

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

    async def check_connection(self):
        """Check if the connection to the Deriv API is active."""
        try:
            api = await self._get_api_instance()
            # Ping the API with a simple request
            response = await api.ping()
            self.is_connected = "ping" in response and response["ping"] == "pong"
            return self.is_connected
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            self.is_connected = False
            return False

    async def get_account_info(self):
        """Get account information including balance."""
        if not self.api_token:
            logger.warning("API token is required to get account information")
            return None

        try:
            api = await self._get_api_instance()
            response = await api.authorize({"authorize": self.api_token})

            if "error" in response:
                logger.error(f"Error getting account info: {response['error']['message']}")
                self.is_connected = False
                return None

            self.is_connected = True
            self.account_info = response.get("authorize", {})
            return self.account_info
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            self.is_connected = False
            return None

    async def get_account_balance(self):
        """Get the account balance."""
        if self.account_info is None:
            await self.get_account_info()

        if self.account_info:
            balance = self.account_info.get("balance")
            currency = self.account_info.get("currency")
            if balance is not None and currency:
                return {"balance": balance, "currency": currency}

        return None

    async def get_available_durations(self, symbol, contract_type):
        """
        Get available durations for a specific symbol and contract type.

        Args:
            symbol: Market symbol
            contract_type: Type of contract (CALL/PUT)

        Returns:
            List of available durations in seconds, or None if error
        """
        if not symbol:
            logger.error("Symbol is required to get available durations")
            return None

        try:
            api = await self._get_api_instance()

            # Explicitly authorize with the API token
            if self.api_token:
                auth_response = await api.authorize({"authorize": self.api_token})

                if "error" in auth_response:
                    error_msg = f"Authorization failed: {auth_response['error']['message']}"
                    logger.error(error_msg)
                    return None

                # Update account info after successful authorization
                self.account_info = auth_response.get("authorize", {})

            # Query contracts_for endpoint to get available contracts
            response = await api.contracts_for({"contracts_for": symbol, "currency": "USD"})

            if "error" in response:
                error_msg = f"Error getting contracts: {response['error']['message']}"
                logger.error(error_msg)
                return None

            # Extract available durations for the specified contract type
            available_durations = []
            contracts = response.get("contracts_for", {}).get("available", [])

            for contract in contracts:
                if contract.get("contract_type") == contract_type:
                    # Get min/max duration
                    min_duration = contract.get("min_contract_duration")
                    max_duration = contract.get("max_contract_duration")

                    if min_duration and max_duration:
                        # Convert duration strings to seconds
                        min_seconds = self._duration_to_seconds(min_duration)
                        max_seconds = self._duration_to_seconds(max_duration)

                        # Generate a list of standard durations within the range
                        standard_durations = [
                            30,
                            60,
                            120,
                            180,
                            300,
                            600,
                            900,
                            1800,
                            3600,
                            7200,
                            14400,
                            28800,
                            86400,
                        ]
                        for duration in standard_durations:
                            if min_seconds <= duration <= max_seconds:
                                available_durations.append(duration)

            # Remove duplicates and sort
            available_durations = sorted(list(set(available_durations)))

            if not available_durations:
                logger.warning(
                    f"No available durations found for {symbol} with contract type {contract_type}",
                )

            return available_durations

        except Exception as e:
            logger.error(f"Error getting available durations: {e}")
            return None

    def _duration_to_seconds(self, duration_str):
        """
        Convert duration string (e.g., "3m", "1h", "1d") to seconds.

        Args:
            duration_str: Duration string in format like "3m", "1h", "1d"

        Returns:
            Duration in seconds
        """
        if not duration_str:
            return 0

        # Extract number and unit
        unit = duration_str[-1].lower()
        value = int(duration_str[:-1])

        # Convert to seconds based on unit
        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            logger.warning(f"Unknown duration unit: {unit}")
            return 0

import asyncio
from deriv_api import DerivAPI

class DerivClient:
    """Client for interacting with the Deriv API."""
    
    def __init__(self, app_id='70789'):
        self.app_id = app_id
        
    async def get_active_symbols(self):
        """Fetch and return active market symbols."""
        try:
            api = DerivAPI(app_id=self.app_id)
            response = await api.active_symbols({"active_symbols": "brief", "product_type": "basic"})
            active_symbols = response.get('active_symbols', [])
            symbols_dict = {f"{symbol['market_display_name']} - {symbol['display_name']}": symbol['symbol'] for
                           symbol in active_symbols}
            return symbols_dict
        except Exception as e:
            print(f"Error fetching active symbols: {e}")
            return {}
            
    async def get_candles(self, symbol, granularity, count=50):
        """Retrieve candle data for a specific symbol and granularity."""
        try:
            api = DerivAPI(app_id=self.app_id)
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

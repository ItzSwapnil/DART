import asyncio
from api.deriv_client import DerivClient
from config.dart_config import get_config

async def test():
    config = get_config()
    client = DerivClient(
        app_id=config.api.deriv_app_id,
        access_token=config.api.deriv_access_token or None,
        account_id=config.api.deriv_account_id or None,
    )
    print("Connecting...")
    connected = await client.check_connection()
    if connected:
        symbols_dict = await client.get_active_symbols()
        print("Total symbols returned by get_active_symbols():", len(symbols_dict))
        
        categories = {
            "Derived": [],
            "Baskets": [],
            "Synthetics": [],
            "Forex": [],
            "Stock Indices": [],
            "Cryptocurrencies": [],
            "Commodities": [],
        }

        def get_category_group(market_display, market_id):
            md = str(market_display).lower()
            mi = str(market_id).lower()
            if "basket" in md or "basket" in mi:
                return "Baskets"
            elif "synthetic" in md or "synthetic" in mi:
                return "Synthetics"
            elif "derived" in md or "derived" in mi:
                return "Derived"
            elif "forex" in md or "forex" in mi:
                return "Forex"
            elif "crypto" in md or "crypto" in mi:
                return "Cryptocurrencies"
            elif "commodity" in md or "commodity" in mi or "metal" in md or "metal" in mi:
                return "Commodities"
            elif "indices" in md or "index" in md or "indices" in mi or "stock" in md or "stock" in mi:
                return "Stock Indices"
            return "Derived"

        for item in client.active_symbols_raw:
            symbol = item.get("symbol") or item.get("underlying_symbol")
            display_name = item.get("display_name") or item.get("underlying_symbol_name") or symbol
            market_name = item.get("market_display_name") or item.get("market") or ""
            market_id = item.get("market") or ""

            name = f"{market_name} - {display_name}" if market_name else str(display_name)
            if not item.get("exchange_is_open", True):
                name = f"{name} [CLOSED]"

            grp = get_category_group(market_name, market_id)
            categories[grp].append(name)
            
        for cat, syms in categories.items():
            print(f"Category '{cat}': {len(syms)} symbols. Sample: {syms[:5] if syms else []}")
            
if __name__ == "__main__":
    asyncio.run(test())

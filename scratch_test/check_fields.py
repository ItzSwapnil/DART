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
        for item in client.active_symbols_raw:
            symbol = item.get("symbol")
            if "silver" in symbol.lower() or "gold" in symbol.lower() or "basket" in symbol.lower() or "r_75" in symbol.lower():
                print(f"Symbol: {symbol} | display_name: {item.get('display_name')} | market: {item.get('market')} | market_display_name: {item.get('market_display_name')} | submarket: {item.get('submarket')} | submarket_display_name: {item.get('submarket_display_name')}")

if __name__ == "__main__":
    asyncio.run(test())

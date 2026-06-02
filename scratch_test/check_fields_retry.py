import asyncio
import logging
from api.deriv_client import DerivClient
from config.dart_config import get_config

logging.basicConfig(level=logging.INFO)

async def test():
    config = get_config()
    client = DerivClient(
        app_id=config.api.deriv_app_id,
        access_token=config.api.deriv_access_token or None,
        account_id=config.api.deriv_account_id or None,
    )
    
    for attempt in range(1, 6):
        print(f"Attempt {attempt} to connect...")
        try:
            connected = await client.check_connection()
            if connected:
                print("Connected successfully!")
                symbols_dict = await client.get_active_symbols()
                print("Total raw symbols:", len(client.active_symbols_raw))
                for item in client.active_symbols_raw:
                    symbol = item.get("symbol", "")
                    if any(x in symbol.lower() for x in ["gold", "silver", "basket", "r_75"]):
                        print(f"Symbol: {symbol} | display_name: {item.get('display_name')} | market: {item.get('market')} | market_display_name: {item.get('market_display_name')} | submarket: {item.get('submarket')} | submarket_display_name: {item.get('submarket_display_name')}")
                break
        except Exception as e:
            print("Error:", e)
        await asyncio.sleep(2)
    
    await client.disconnect() if hasattr(client, 'disconnect') else None

if __name__ == "__main__":
    asyncio.run(test())

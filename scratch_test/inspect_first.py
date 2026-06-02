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
    for attempt in range(1, 6):
        print(f"Attempt {attempt}...")
        try:
            connected = await client.check_connection()
            if connected:
                await client.get_active_symbols()
                if client.active_symbols_raw:
                    print("Keys of first item:", list(client.active_symbols_raw[0].keys()))
                    print("First item content:", client.active_symbols_raw[0])
                    print("All symbol names:")
                    for item in client.active_symbols_raw:
                        print(f"symbol: {item.get('symbol')} | display_name: {item.get('display_name')} | market: {item.get('market')} | market_display_name: {item.get('market_display_name')}")
                    break
        except Exception as e:
            print("Error:", e)
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test())

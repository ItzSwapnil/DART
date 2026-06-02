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
    print("Connected:", connected)
    if connected:
        symbols = await client.get_active_symbols()
        print("Total symbols returned by get_active_symbols():", len(symbols))
        print("First 10 symbols:", list(symbols.keys())[:10])
        # Let's count how many categorized
        from ui.modern_dashboard import DASHBOARD_STATE
        print("Initial state available markets count:", len(DASHBOARD_STATE["available_markets"]))
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test())

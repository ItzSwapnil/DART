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
    print("Fetching active symbols via client...")
    symbols = await client.get_active_symbols()
    print(f"Total symbols parsed: {len(symbols)}")
    print("First 5 symbols:")
    for k, v in list(symbols.items())[:5]:
        print(f"  {k} => {v}")


if __name__ == "__main__":
    asyncio.run(test())

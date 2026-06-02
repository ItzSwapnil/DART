import asyncio
import json
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
        try:
            connected = await client.check_connection()
            if connected:
                resp = await client._send_and_receive_public(
                    {"active_symbols": "brief", "req_id": 1},
                    match_key="active_symbols",
                )
                raw = resp.get("active_symbols", [])
                print(f"Total raw symbols in response: {len(raw)}")
                if raw:
                    print("Sample item:", json.dumps(raw[0], indent=2))
                break
        except Exception as e:
            print("Error:", e)
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test())

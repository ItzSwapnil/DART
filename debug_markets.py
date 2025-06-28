#!/usr/bin/env python3
"""Debug script to test market fetching."""

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.deriv_client import DerivClient
from config.settings import DERIV_APP_ID, DERIV_API_TOKEN

async def test_markets():
    """Test market fetching."""
    print("Testing market fetching...")
    
    client = DerivClient(app_id=DERIV_APP_ID, api_token=DERIV_API_TOKEN)
    
    try:
        markets = await client.get_active_symbols()
        print(f"Market type: {type(markets)}")
        print(f"Market count: {len(markets) if markets else 0}")
        
        if markets:
            print("\nFirst 10 markets:")
            for i, (name, symbol) in enumerate(list(markets.items())[:10]):
                print(f"  {i+1}. {name} -> {symbol}")
            
            print(f"\nTotal markets available: {len(markets)}")
            
            # Show some specific categories
            forex_markets = [name for name in markets.keys() if 'USD' in name or 'EUR' in name or 'GBP' in name]
            crypto_markets = [name for name in markets.keys() if any(crypto in name.lower() for crypto in ['bitcoin', 'ethereum', 'crypto'])]
            volatility_markets = [name for name in markets.keys() if 'volatility' in name.lower() or 'boom' in name.lower() or 'crash' in name.lower()]
            
            print(f"\nCategories:")
            print(f"  Forex markets: {len(forex_markets)}")
            print(f"  Crypto markets: {len(crypto_markets)}")
            print(f"  Volatility indices: {len(volatility_markets)}")
            
        else:
            print("No markets returned!")
            
    except Exception as e:
        print(f"Error fetching markets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_markets())

#!/usr/bin/env python3
"""Debug script to test model training."""

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.deriv_client import DerivClient
from ml.trading_ai import TradingAI
from ml.auto_trader import AutoTrader
from config.settings import DERIV_APP_ID, DERIV_API_TOKEN

async def test_training():
    """Test model training."""
    print("Testing model training...")
    
    try:
        # Initialize components
        client = DerivClient(app_id=DERIV_APP_ID, api_token=DERIV_API_TOKEN)
        trading_ai = TradingAI(
            model_dir=os.path.join(os.path.dirname(__file__), 'models'),
            use_deep_rl=False,  # Start with basic
            use_enhanced_features=False
        )
        auto_trader = AutoTrader(client=client, trading_ai=trading_ai)
        
        # Test with a common symbol
        symbol = "frxEURUSD"
        granularity = 60  # 1 minute
        
        print(f"Testing training for symbol: {symbol}, granularity: {granularity}")
        
        # Get historical data first
        print("Fetching historical data...")
        historical_data = await client.get_historical_data(symbol, granularity, days=7)
        
        if not historical_data:
            print("❌ No historical data returned")
            return
        
        print(f"✅ Historical data fetched: {len(historical_data)} candles")
        
        # Test direct training
        print("Testing direct TradingAI training...")
        metrics = trading_ai.train_model(historical_data)
        
        if metrics:
            print("✅ Direct training successful!")
            print(f"Metrics: {metrics}")
        else:
            print("❌ Direct training failed")
        
        # Test auto-trader training
        print("Testing AutoTrader training...")
        success = await auto_trader.train_model(symbol, granularity)
        
        if success:
            print("✅ AutoTrader training successful!")
        else:
            print("❌ AutoTrader training failed")
            
    except Exception as e:
        print(f"❌ Error during training test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_training())

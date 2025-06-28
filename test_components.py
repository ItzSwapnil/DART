#!/usr/bin/env python3
"""Quick test to verify DART components are working."""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing DART Component Imports...")
    
    try:
        # Test core components
        from api.deriv_client import DerivClient
        print("✅ DerivClient imported successfully")
        
        from ml.trading_ai import TradingAI
        print("✅ TradingAI imported successfully")
        
        from ml.auto_trader import AutoTrader
        print("✅ AutoTrader imported successfully")
        
        from config.settings import DERIV_APP_ID, DERIV_API_TOKEN
        print("✅ Settings imported successfully")
        
        # Test UI components
        from ui.app import DerivApp
        print("✅ Desktop UI imported successfully")
        
        from dart_launcher_new import DARTLauncher
        print("✅ Launcher imported successfully")
        
        # Test modern dashboard (might fail if streamlit not available)
        try:
            import streamlit as st
            print("✅ Streamlit available for web dashboard")
        except ImportError:
            print("⚠️ Streamlit not available (web dashboard may not work)")
        
        print("\n🎉 All core components imported successfully!")
        print("🚀 DART is ready to launch!")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✨ You can now run DART with: uv run python main.py")
    else:
        print("\n💥 Please fix the import issues before running DART")
        sys.exit(1)

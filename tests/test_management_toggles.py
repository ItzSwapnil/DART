"""Tests for risk, money, and time management toggles in AutoTrader."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from ml.auto_trader import AutoTrader
from ml.trading_ai_v3 import TradingAIv3


@pytest.mark.asyncio
async def test_auto_money_management_bypass() -> None:
    """Test that setting auto_money_management=False bypasses adjustments and uses base_trade_amount."""
    client = MagicMock()
    trading_ai = MagicMock()
    trader = AutoTrader(client=client, trading_ai=trading_ai)

    trader.auto_money_management = False
    trader.base_trade_amount = 25.5
    trader.current_trade_amount = 10.0

    # Mock execute_trade to inspect the amount used
    trader.execute_trade = AsyncMock()

    strategy = {
        "contract_type": "CALL",
        "direction": "CALL",
        "duration": 60,
        "duration_unit": "s",
        "confidence": 0.9,
        "recommended_amount": 50.0  # Risk manager recommended $50
    }

    # Run trading cycle
    with patch.object(trader, "generate_strategy", AsyncMock(return_value=strategy)), \
         patch.object(trader, "check_trading_conditions", MagicMock(return_value=True)), \
         patch.object(trader, "symbol_last_training", {"R_75": "some_time"}):

        await trader.trading_cycle(symbol="R_75", granularity=60)

    # Verify that the trade amount used is exactly the base_trade_amount (25.5), NOT recommended_amount (50.0)
    trader.execute_trade.assert_called_once()
    kwargs = trader.execute_trade.call_args[1]
    assert kwargs["amount"] == 25.5


@pytest.mark.asyncio
async def test_auto_time_management_bypass() -> None:
    """Test that setting auto_time_management=False overrides duration with manual_duration."""
    client = MagicMock()
    trading_ai = MagicMock()
    trader = AutoTrader(client=client, trading_ai=trading_ai)

    trader.auto_time_management = False
    trader.manual_duration = 120
    trader.manual_duration_unit = "m"

    trader.execute_trade = AsyncMock()

    strategy = {
        "contract_type": "CALL",
        "direction": "CALL",
        "duration": 30,
        "duration_unit": "s",
        "confidence": 0.8
    }

    with patch.object(trader, "generate_strategy", AsyncMock(return_value=strategy)), \
         patch.object(trader, "check_trading_conditions", MagicMock(return_value=True)), \
         patch.object(trader, "symbol_last_training", {"R_75": "some_time"}):

        await trader.trading_cycle(symbol="R_75", granularity=60)

    # Verify that the duration is overridden to the manual values
    trader.execute_trade.assert_called_once()
    kwargs = trader.execute_trade.call_args[1]
    assert kwargs["strategy"]["duration"] == 120
    assert kwargs["strategy"]["duration_unit"] == "m"


@pytest.mark.asyncio
async def test_risk_management_bypass() -> None:
    """Test that setting risk_management_enabled=False bypasses risk blocks and daily limits."""
    client = MagicMock()
    trading_ai = MagicMock()
    trader = AutoTrader(client=client, trading_ai=trading_ai)

    trader.risk_management_enabled = False
    # Set daily loss over the limit
    trader.daily_loss = 1000.0

    trader.execute_trade = AsyncMock()

    strategy = {
        "contract_type": "CALL",
        "direction": "CALL",
        "duration": 60,
        "duration_unit": "s",
        "confidence": 0.85,
        "risk_blocked": True,
        "risk_reason": "High Drawdown"
    }

    with patch.object(trader, "generate_strategy", AsyncMock(return_value=strategy)), \
         patch.object(trader, "check_trading_conditions", MagicMock(return_value=True)), \
         patch.object(trader, "symbol_last_training", {"R_75": "some_time"}):

        await trader.trading_cycle(symbol="R_75", granularity=60)

    # Verify that execute_trade was called because risk checks were bypassed
    trader.execute_trade.assert_called_once()


@pytest.mark.asyncio
async def test_max_simultaneous_trades_configuration() -> None:
    """Test that max_simultaneous_trades is configured via start_trading."""
    client = MagicMock()
    trading_ai = MagicMock()
    trader = AutoTrader(client=client, trading_ai=trading_ai)
    
    # Default value should be 5
    assert trader.max_simultaneous_trades == 5
    
    # Update via start_trading
    with patch.object(trader, "_run_trading_thread", MagicMock()):
        trader.start_trading(
            symbol="R_75",
            granularity=60,
            max_simultaneous_trades=8
        )
        assert trader.max_simultaneous_trades == 8


@pytest.mark.asyncio
async def test_get_account_balance_force_refresh() -> None:
    """Test that get_account_balance force argument triggers get_account_info refresh."""
    client = MagicMock()
    client.access_token = "pat_token"
    client.account_info = None
    client._is_pat = MagicMock(return_value=True)
    client._trading_ws = MagicMock()
    client._get_trading_ws = AsyncMock(return_value=client._trading_ws)
    client.http_timeout = 20
    
    # Mock send and receive for PAT authorize call
    auth_response = {
        "authorize": {
            "loginid": "CR12345",
            "balance": "9876.54",
            "currency": "USD",
            "email": "test@deriv.com",
            "fullname": "Test User",
            "is_virtual": 1
        }
    }
    client._send_and_receive = AsyncMock(return_value=auth_response)
    
    # Bind the actual methods to check they run correctly
    from api.deriv_client import DerivClient
    client.get_account_info = DerivClient.get_account_info.__get__(client, DerivClient)
    client.get_account_balance = DerivClient.get_account_balance.__get__(client, DerivClient)
    
    # Initial balance call
    res = await client.get_account_balance(force=True)
    assert res["balance"] == "9876.54"
    assert res["currency"] == "USD"
    
    # Verify authorize was called on WebSocket
    client._send_and_receive.assert_called_once()


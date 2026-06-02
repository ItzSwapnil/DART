"""Tests for portfolio management and WebSocket server."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.websocket_server import DARTWebSocketServer
from ml.auto_trader import AutoTrader
from ml.trading_ai_v3 import TradingAIv3


@pytest.mark.asyncio
async def test_websocket_server_broadcast() -> None:
    """Test starting the DART WebSocket server and broadcasting messages to a client."""
    # Use ephemeral port
    server = DARTWebSocketServer(host="127.0.0.1", port=18765)
    started = await server.start()
    assert started is True

    import websockets

    # Connect client
    uri = "ws://127.0.0.1:18765"
    async with websockets.connect(uri) as client:
        # Give registration a split second
        await asyncio.sleep(0.1)

        # Broadcast test message
        test_msg = {"status": "info", "message": "Test broadcast message"}
        server.broadcast(test_msg)

        # Receive on client side
        response = await asyncio.wait_for(client.recv(), timeout=2.0)
        data = json.loads(response)
        assert data["message"] == "Test broadcast message"
        assert data["status"] == "info"

    # Stop server
    await server.stop()


def test_autotrader_multi_symbol_parsing() -> None:
    """Test that AutoTrader correctly parses multi-symbol strings and lists."""
    client = MagicMock()
    client.check_all_trades = AsyncMock()
    trading_ai = MagicMock()

    trader = AutoTrader(client=client, trading_ai=trading_ai)

    # Mock WebSocket server to avoid running it during parsing test
    mock_ws = MagicMock()
    # Side effect to stop loop immediately before it enters the while loop
    mock_ws.start = AsyncMock(side_effect=lambda: setattr(trader, "is_running", False))
    mock_ws.stop = AsyncMock()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    with patch("ml.auto_trader.DARTWebSocketServer", return_value=mock_ws), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        async def run_parse_test():
            await trader.run_trading_loop(symbol="R_75, R_50, frxEURUSD", granularity=60)

        loop.run_until_complete(run_parse_test())

    assert trader.symbols == ["R_75", "R_50", "frxEURUSD"]
    assert trader.current_symbol in trader.symbols
    assert trader.current_granularity == 60


def test_autotrader_symbol_specific_models() -> None:
    """Test that AutoTrader instantiates distinct models for different symbols."""
    client = MagicMock()
    trading_ai = MagicMock()

    trader = AutoTrader(client=client, trading_ai=trading_ai)

    model_a = trader.get_model_for_symbol("R_75")
    model_b = trader.get_model_for_symbol("R_50")

    assert model_a is not model_b
    assert isinstance(model_a, TradingAIv3)
    assert isinstance(model_b, TradingAIv3)


@pytest.mark.asyncio
async def test_risk_manager_updated_on_execution() -> None:
    """Test that AdvancedRiskManager is updated with size > 0 when a trade is executed."""
    client = MagicMock()
    client.buy_contract = AsyncMock(return_value={"buy": {"contract_id": 12345, "ask_price": 10.0}})

    trading_ai = MagicMock()
    trader = AutoTrader(client=client, trading_ai=trading_ai)

    # Mock Risk Manager
    risk_manager = MagicMock()
    trader.risk_manager = risk_manager

    strategy = {"contract_type": "CALL", "direction": "CALL", "duration": 60, "duration_unit": "s", "confidence": 0.85}

    contract_id = await trader.execute_trade(symbol="R_75", strategy=strategy, amount=10.0, price=100.0)

    assert contract_id == 12345
    # Verify update_position was called with correct size/price
    risk_manager.update_position.assert_called_once_with(symbol="R_75", size=10.0, price=1.0)

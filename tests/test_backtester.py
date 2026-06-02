"""Unit tests for the Backtesting Framework."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ml.backtester import Backtester
from ml.trading_ai_v3 import FeatureEngineer, TradingAIv3


@pytest.fixture
def synthetic_market_data() -> pd.DataFrame:
    """Generate 150 points of synthetic OHLCV data."""
    np.random.seed(42)
    n_points = 150
    dates = pd.date_range(start="2026-05-01", periods=n_points, freq="min")
    close = 100.0 + np.cumsum(np.random.normal(0, 0.2, n_points))
    open_p = close + np.random.normal(0, 0.05, n_points)
    high = np.maximum(open_p, close) + np.random.exponential(0.02, n_points)
    low = np.minimum(open_p, close) - np.random.exponential(0.02, n_points)
    volume = np.random.randint(100, 1000, n_points)

    return pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_backtester_initialization() -> None:
    """Test that Backtester initializes with default and custom values."""
    mock_model = MagicMock(spec=TradingAIv3)
    bt = Backtester(model=mock_model, initial_capital=5000.0, win_payout_ratio=0.85)

    assert bt.model == mock_model
    assert bt.initial_capital == 5000.0
    assert bt.win_payout_ratio == 0.85
    assert bt.loss_ratio == -1.0
    assert bt.risk_manager.current_capital == 5000.0


def test_backtester_run_simulation(synthetic_market_data: pd.DataFrame) -> None:
    """Test backtesting run method using mocked signals to ensure correct flow."""
    mock_model = MagicMock(spec=TradingAIv3)
    # Ensure model.ensemble_model check passes
    mock_model.ensemble_model = MagicMock()
    mock_model.feature_engineer = FeatureEngineer()

    def mock_generate_strategy(recent_data: list[dict], use_llm: bool = False) -> dict | None:
        # Check current position/index in the dataset by looking at the last element's values
        if not recent_data:
            return None

        # Generate predictable signals
        # We trigger signal when the data length matches specific step values in the loop
        # Loop starts at start_pos (e.g. 20) and goes to len(data) (150)
        # So len(recent_data) will go from 21 to 150
        current_len = len(recent_data)

        if current_len == 40:
            return {
                "direction": "CALL",
                "confidence": 0.8,
                "volatility": 0.02,
                "duration": 60,  # 1 step in 1-min candles
            }
        elif current_len == 80:
            return {
                "direction": "PUT",
                "confidence": 0.7,
                "volatility": 0.02,
                "duration": 60,  # 1 step in 1-min candles
            }
        return None

    mock_model.generate_strategy.side_effect = mock_generate_strategy

    # Initialize backtester
    bt = Backtester(
        model=mock_model,
        initial_capital=1000.0,
        win_payout_ratio=0.80,
        loss_ratio=-1.0,
        min_trade_amount=10.0,
        max_trade_amount=100.0,
    )

    report = bt.run(synthetic_market_data, allow_concurrent=False)

    # Verify report keys
    assert "initial_capital" in report
    assert "final_capital" in report
    assert "net_profit" in report
    assert "total_trades" in report
    assert "win_rate" in report
    assert "win_count" in report
    assert "loss_count" in report
    assert "profit_factor" in report
    assert "max_drawdown" in report
    assert "sharpe_ratio" in report
    assert "sortino_ratio" in report
    assert "beta" in report
    assert "alpha" in report
    assert "trades" in report

    # Verify at least the mocked trades were registered
    assert report["total_trades"] > 0
    assert len(report["trades"]) == report["total_trades"]

    # Check that final capital matches initial_capital + sum of pnl
    pnl_sum = sum(t["pnl"] for t in report["trades"])
    assert report["final_capital"] == pytest.approx(report["initial_capital"] + pnl_sum)


def test_backtester_untrained_model_error(synthetic_market_data: pd.DataFrame) -> None:
    """Test that run raises error if model is untrained."""
    mock_model = MagicMock(spec=TradingAIv3)
    mock_model.ensemble_model = None

    bt = Backtester(model=mock_model)

    with pytest.raises(ValueError, match="Model must be trained"):
        bt.run(synthetic_market_data)

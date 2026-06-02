"""Tests for the advanced risk management layer."""

import numpy as np
import pytest

from ml.risk_manager import AdvancedRiskManager


@pytest.fixture
def risk_manager() -> AdvancedRiskManager:
    """Fixture to initialize a risk manager with default capital."""
    return AdvancedRiskManager(initial_capital=10000)


def test_calculate_base_position_size(risk_manager: AdvancedRiskManager) -> None:
    """Test base position sizing adjusts dynamically to signal and volatility."""
    # Positive signal, moderate volatility
    size_pos = risk_manager._calculate_base_position_size(
        signal_strength=0.8, confidence=0.9, expected_return=0.02, expected_volatility=0.10
    )
    assert size_pos > 0
    assert size_pos <= risk_manager.position_limits.max_position_size

    # Negative signal
    size_neg = risk_manager._calculate_base_position_size(
        signal_strength=-0.5, confidence=0.8, expected_return=-0.01, expected_volatility=0.08
    )
    assert size_neg < 0


def test_calculate_position_size_interface(risk_manager: AdvancedRiskManager) -> None:
    """Test position sizing recommendations dictionary interface."""
    res = risk_manager.calculate_position_size(
        signal_strength=0.7,
        confidence=0.8,
        expected_return=0.03,
        expected_volatility=0.12,
        symbol="R_75",
        current_price=150.0,
    )

    assert "recommended_size" in res
    assert "base_size" in res
    assert "kelly_size" in res
    assert "vol_target_size" in res
    assert "max_loss" in res
    assert res["recommended_size"] <= risk_manager.position_limits.max_position_size


def test_circuit_breakers(risk_manager: AdvancedRiskManager) -> None:
    """Test that daily loss and drawdown alerts trigger halts correctly."""
    # Under normal conditions, should not halt
    risk_manager.daily_pnl.append(0.0)
    risk_manager.drawdown_history.append(0.0)
    check_normal = risk_manager.circuit_breaker_check()
    assert check_normal["should_halt"] is False

    # Simulate 6% daily loss (exceeds 5% halt threshold)
    risk_manager.daily_pnl.append(-600.0)
    check_loss = risk_manager.circuit_breaker_check()
    assert check_loss["should_halt"] is True
    assert any(alert["type"] == "daily_loss_halt" for alert in check_loss["alerts"])

    # Simulate 16% drawdown (exceeds 15% halt threshold)
    risk_manager.daily_pnl.clear()
    risk_manager.drawdown_history.append(-0.16)
    check_drawdown = risk_manager.circuit_breaker_check()
    assert check_drawdown["should_halt"] is True
    assert any(alert["type"] == "drawdown_halt" for alert in check_drawdown["alerts"])


def test_monte_carlo_var(risk_manager: AdvancedRiskManager) -> None:
    """Test Monte Carlo VaR computation."""
    # Mock returns: 100 days of slightly positive returns
    np.random.seed(42)
    mock_returns = np.random.normal(0.001, 0.01, 100)

    var_95, var_99 = risk_manager._calculate_monte_carlo_var(mock_returns)

    assert var_95 < 0
    assert var_99 < 0
    # 99% VaR should be more negative than 95% VaR
    assert var_99 < var_95


def test_run_stress_tests(risk_manager: AdvancedRiskManager) -> None:
    """Test portfolio stress testing scenario execution."""
    # Setup some open positions
    risk_manager.current_positions = {"R_75": {"size": 2.0, "price": 100.0}, "R_100": {"size": -1.0, "price": 200.0}}

    stress_res = risk_manager.run_stress_tests()

    assert "market_crash" in stress_res
    assert "flash_crash" in stress_res
    assert "absolute_impact" in stress_res["market_crash"]
    assert "severity" in stress_res["market_crash"]

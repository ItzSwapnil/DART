"""
Backtesting Framework for DART.
Vectorized and event-driven simulation of options contracts using model signals and risk management.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.risk_manager import AdvancedRiskManager
from ml.trading_ai_v3 import TradingAIv3

logger = logging.getLogger("backtester")


class Backtester:
    """Simulates historical trading using TradingAIv3 and AdvancedRiskManager."""

    def __init__(
        self,
        model: TradingAIv3,
        initial_capital: float = 10000.0,
        win_payout_ratio: float = 0.80,
        loss_ratio: float = -1.0,
        candle_interval_seconds: int = 60,
        min_trade_amount: float = 1.0,
        max_trade_amount: float = 1000.0,
    ):
        """
        Initialize the Backtester.

        Args:
            model: Trained TradingAIv3 instance
            initial_capital: Starting capital for the backtest
            win_payout_ratio: Payout ratio on a winning trade (e.g. 0.80 for 80% return)
            loss_ratio: Loss ratio on a losing trade (e.g. -1.0 for 100% loss of stake)
            candle_interval_seconds: Default duration of a single candle in seconds
            min_trade_amount: Minimum stake size allowed
            max_trade_amount: Maximum stake size allowed
        """
        self.model = model
        self.initial_capital = initial_capital
        self.win_payout_ratio = win_payout_ratio
        self.loss_ratio = loss_ratio
        self.candle_interval_seconds = candle_interval_seconds
        self.min_trade_amount = min_trade_amount
        self.max_trade_amount = max_trade_amount

        # Initialize risk manager with same initial capital
        self.risk_manager = AdvancedRiskManager(initial_capital=self.initial_capital)

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "R_75",
        allow_concurrent: bool = False,
        lookback_limit: int = 200,
    ) -> dict[str, Any]:
        """
        Run backtest over historical OHLCV data.

        Args:
            data: DataFrame containing OHLCV columns (open, high, low, close)
            symbol: Symbol to simulate trading for
            allow_concurrent: Allow opening new trades while others are active
            lookback_limit: Maximum number of historical candles to feed to the model at each step

        Returns:
            Backtest report summary dictionary
        """
        if self.model.ensemble_model is None:
            raise ValueError("Model must be trained before running backtests.")

        # Standardize columns to lowercase
        data = data.copy()
        data.columns = [col.lower() for col in data.columns]
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Data is missing required columns: {required_cols}")

        # Attempt to auto-detect candle interval if index is DatetimeIndex
        interval = self.candle_interval_seconds
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            diffs = pd.Series(data.index).diff().dropna()
            if not diffs.empty:
                interval = int(diffs.median().total_seconds())
                logger.info(f"Auto-detected candle interval: {interval} seconds")

        # Get features to determine the index where indicators are valid
        features_df = self.model.feature_engineer.engineer_features(data)
        if features_df.empty:
            raise ValueError("Feature engineering returned empty DataFrame. Insufficient data.")

        start_idx = features_df.index[0]
        # Find integer index of the start
        if isinstance(data.index, pd.DatetimeIndex):  # noqa: SIM108
            start_pos = int(data.index.get_loc(start_idx))
        else:
            start_pos = int(start_idx)

        current_balance = self.initial_capital
        active_trades: list[dict[str, Any]] = []
        trade_log = []

        # Reset risk manager
        self.risk_manager = AdvancedRiskManager(initial_capital=self.initial_capital)

        logger.info(f"Starting backtest from index/position {start_pos} with capital {current_balance}...")

        # Loop chronologically through the data
        for pos in range(start_pos, len(data)):
            current_time = data.index[pos]
            current_price = float(data["close"].iloc[pos])

            # 1. Resolve expired trades
            resolved_trades = []
            for trade in active_trades:
                if pos >= trade["exit_pos"]:
                    # Determine winner
                    exit_price = float(data["close"].iloc[pos])
                    entry_price = trade["entry_price"]
                    direction = trade["direction"]

                    won = exit_price > entry_price if direction == "CALL" else exit_price < entry_price

                    trade_return = self.win_payout_ratio if won else self.loss_ratio
                    pnl = trade_return * trade["stake"]
                    current_balance += pnl

                    # Update risk manager
                    benchmark_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

                    # Log return and update portfolio values (size=stake triggers capital change)
                    self.risk_manager.update_position(
                        symbol=symbol,
                        size=trade["stake"],
                        price=exit_price,
                        trade_return=trade_return,
                        benchmark_return=benchmark_return,
                    )

                    # Close position in risk manager
                    self.risk_manager.update_position(
                        symbol=symbol,
                        size=0.0,
                        price=exit_price,
                    )

                    trade_log.append(
                        {
                            "entry_time": trade["entry_time"],
                            "exit_time": current_time,
                            "direction": direction,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "stake": trade["stake"],
                            "won": won,
                            "pnl": pnl,
                            "balance": current_balance,
                        }
                    )
                    resolved_trades.append(trade)

            # Remove resolved trades
            for trade in resolved_trades:
                active_trades.remove(trade)

            # Check if bankrupted
            if current_balance < self.min_trade_amount:
                logger.warning("Capital fell below minimum trade amount. Stopping backtest.")
                break

            # 2. Enter new trade if conditions met
            if not allow_concurrent and len(active_trades) > 0:
                continue

            # Circuit breaker/Risk checks
            cb_check = self.risk_manager.circuit_breaker_check()
            violations = self.risk_manager.check_risk_limits()
            if cb_check.get("should_halt", False) or any(v.get("severity") == "high" for v in violations):
                continue

            # Slice recent data for the model
            slice_start = max(0, pos - lookback_limit + 1)
            recent_df = data.iloc[slice_start : pos + 1]
            recent_data_list = recent_df.to_dict("records")

            # Generate strategy signal
            strategy = self.model.generate_strategy(recent_data_list, use_llm=False)
            if strategy is None or strategy.get("direction") not in ["CALL", "PUT"]:
                continue

            direction = strategy["direction"]
            confidence = strategy["confidence"]
            volatility = strategy.get("volatility", 0.02)
            duration = strategy.get("duration", 60)

            # Calculate step count for duration
            steps = max(1, round(duration / interval))
            exit_pos = pos + steps

            # Check if exit is out of bounds
            if exit_pos >= len(data):
                continue

            # Sizing from Risk Manager
            signal_strength = 1.0 if direction == "CALL" else -1.0
            expected_return = signal_strength * confidence * volatility

            risk_recommendation = self.risk_manager.calculate_position_size(
                signal_strength=signal_strength,
                confidence=confidence,
                expected_return=expected_return,
                expected_volatility=volatility,
                symbol=symbol,
                current_price=current_price,
            )

            recommended_amount = abs(risk_recommendation["capital_allocation"])
            stake = max(self.min_trade_amount, min(recommended_amount, self.max_trade_amount))

            # Check if we have enough balance
            if stake > current_balance:
                stake = current_balance

            if stake < self.min_trade_amount:
                continue

            # Open trade in risk manager
            self.risk_manager.update_position(
                symbol=symbol,
                size=stake,
                price=current_price,
            )

            # Record active trade
            active_trades.append(
                {
                    "entry_pos": pos,
                    "exit_pos": exit_pos,
                    "entry_time": current_time,
                    "entry_price": current_price,
                    "direction": direction,
                    "stake": stake,
                }
            )

        # Calculate final metrics
        total_trades = len(trade_log)
        wins = [t for t in trade_log if t["won"]]
        losses = [t for t in trade_log if not t["won"]]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        net_profit = current_balance - self.initial_capital

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe / Sortino using portfolio risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics()

        summary = {
            "initial_capital": self.initial_capital,
            "final_capital": current_balance,
            "net_profit": net_profit,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "win_count": win_count,
            "loss_count": loss_count,
            "profit_factor": profit_factor,
            "max_drawdown": float(risk_metrics.maximum_drawdown),
            "sharpe_ratio": float(risk_metrics.sharpe_ratio),
            "sortino_ratio": float(risk_metrics.sortino_ratio),
            "beta": float(risk_metrics.beta),
            "alpha": float(risk_metrics.alpha),
            "trades": trade_log,
        }

        return summary


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Generate synthetic historical data for demo/testing
    np.random.seed(42)
    n_points = 500
    dates = pd.date_range(start="2026-05-01", periods=n_points, freq="min")
    close = 100.0 + np.cumsum(np.random.normal(0, 0.5, n_points))
    open_p = close + np.random.normal(0, 0.2, n_points)
    high = np.maximum(open_p, close) + np.random.exponential(0.1, n_points)
    low = np.minimum(open_p, close) - np.random.exponential(0.1, n_points)
    volume = np.random.randint(100, 1000, n_points)

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

    logger.info("Initializing models for dry run...")
    ai = TradingAIv3(use_llm=False, require_real_llm=False)

    # Train the model on first 350 points
    train_data = df.iloc[:350].to_dict("records")
    ai.train(train_data)

    # Initialize Backtester
    backtester = Backtester(model=ai, initial_capital=1000.0)

    # Run backtest on the remaining 150 points
    test_data = df.iloc[200:]  # Overlap a bit to ensure feature engineering has enough history
    report = backtester.run(test_data)

    print("\n" + "=" * 50)
    print(" DART BACKTESTER DRY RUN SUMMARY ")
    print("=" * 50)
    print(f"Initial Capital: ${report['initial_capital']:.2f}")
    print(f"Final Capital:   ${report['final_capital']:.2f}")
    print(f"Net Profit:      ${report['net_profit']:.2f}")
    print(f"Total Trades:    {report['total_trades']}")
    print(f"Win Rate:        {report['win_rate']:.2%}")
    print(f"Profit Factor:   {report['profit_factor']:.2f}")
    print(f"Max Drawdown:    {report['max_drawdown']:.2%}")
    print(f"Sharpe Ratio:    {report['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:   {report['sortino_ratio']:.2f}")
    print(f"Beta:            {report['beta']:.2f}")
    print(f"Alpha:           {report['alpha']:.2f}")
    print("=" * 50 + "\n")

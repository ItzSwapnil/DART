"""
SOTA Evaluation and Monitoring Pipeline for DART v3.0
Comprehensive metrics, monitoring, and continuous evaluation.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

logger = logging.getLogger("evaluation_monitor")


@dataclass
class TradeMetrics:
    """Comprehensive trade metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_loss_per_trade: float = 0.0
    risk_reward_ratio: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    log_loss: float = 0.0
    prediction_calibration: float = 0.0
    inference_time_ms: float = 0.0
    model_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class EvaluationPipeline:
    """SOTA evaluation pipeline for model and trading performance."""

    def __init__(
        self,
        project_name: str = "DART-v3",
        entity: str = "dart-trading",
        use_wandb: bool = False,
    ):
        """Initialize evaluation pipeline."""
        self.project_name = project_name
        self.entity = entity
        self.use_wandb = use_wandb
        self.metrics_history: List[TradeMetrics] = []
        self.model_metrics_history: List[ModelMetrics] = []

        if use_wandb and wandb is not None:
            try:
                wandb.init(project=project_name, entity=entity)
                logger.info("Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
        elif use_wandb:
            logger.warning("Weights & Biases requested but package is not installed")
            self.use_wandb = False

    def evaluate_model_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "ensemble",
    ) -> ModelMetrics:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model

        Returns:
            ModelMetrics object
        """
        metrics = ModelMetrics(
            accuracy=float(np.mean(y_pred == y_true)),
            precision=float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else 0.0,
            recall=float(
                np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
                if np.sum(y_true == 1) > 0
                else 0.0
            ),
            f1=float(
                f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
            ),
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )

        # AUC-ROC if probabilities available
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics.auc_roc = float(roc_auc_score(y_true, y_pred_proba))
            except Exception as e:
                logger.debug(f"AUC calculation failed: {e}")

        self.model_metrics_history.append(metrics)

        if self.use_wandb and wandb is not None:
            wandb.log(metrics.to_dict())

        return metrics

    def evaluate_trading_performance(
        self,
        trades: List[Dict],
        initial_balance: float = 1000.0,
    ) -> TradeMetrics:
        """
        Evaluate trading performance.

        Args:
            trades: List of trade dictionaries with 'profit' key
            initial_balance: Starting balance

        Returns:
            TradeMetrics object
        """
        if not trades:
            return TradeMetrics(timestamp=datetime.now().isoformat())

        profits = [t.get("profit", 0) for t in trades]
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)

        total_profit = sum(p for p in profits if p > 0)
        total_loss = abs(sum(p for p in profits if p < 0))

        metrics = TradeMetrics(
            total_trades=len(trades),
            winning_trades=wins,
            losing_trades=losses,
            total_profit=total_profit,
            total_loss=total_loss,
            win_rate=wins / len(trades) if trades else 0.0,
            profit_factor=total_profit / total_loss if total_loss > 0 else 0.0,
            avg_profit_per_trade=total_profit / wins if wins > 0 else 0.0,
            avg_loss_per_trade=total_loss / losses if losses > 0 else 0.0,
            risk_reward_ratio=(
                (total_profit / wins) / (total_loss / losses)
                if wins > 0 and losses > 0
                else 0.0
            ),
            timestamp=datetime.now().isoformat(),
        )

        # Calculate Sharpe ratio
        if len(profits) > 1:
            returns = np.array(profits) / initial_balance
            if np.std(returns) > 0:
                metrics.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        # Calculate max drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics.max_drawdown = float(np.max(drawdown) / initial_balance if len(drawdown) > 0 else 0)

        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for profit in profits:
            if profit > 0:
                consecutive_wins += 1
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                consecutive_wins = 0

        metrics.consecutive_wins = max_consecutive_wins
        metrics.consecutive_losses = max_consecutive_losses

        self.metrics_history.append(metrics)

        if self.use_wandb and wandb is not None:
            wandb.log(metrics.to_dict())

        logger.info(f"Trading Metrics: WR={metrics.win_rate:.1%}, PF={metrics.profit_factor:.2f}")

        return metrics

    def get_performance_summary(self) -> Dict:
        """Get performance summary from history."""
        if not self.metrics_history:
            return {}

        metrics = self.metrics_history[-1]
        return {
            "total_trades": metrics.total_trades,
            "win_rate": f"{metrics.win_rate:.1%}",
            "profit_factor": f"{metrics.profit_factor:.2f}",
            "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{metrics.max_drawdown:.1%}",
            "total_profit": f"${metrics.total_profit:.2f}",
            "risk_reward_ratio": f"{metrics.risk_reward_ratio:.2f}",
        }

    def export_metrics(self, filepath: Path) -> None:
        """Export metrics to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "trade_metrics": [m.to_dict() for m in self.metrics_history],
            "model_metrics": [m.to_dict() for m in self.model_metrics_history],
            "exported_at": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Metrics exported to {filepath}")

    def plot_performance(self, output_dir: Path = Path("./reports")) -> None:
        """Plot performance metrics."""
        try:
            import matplotlib.pyplot as plt

            output_dir.mkdir(parents=True, exist_ok=True)

            if self.metrics_history:
                # Win rate over time
                win_rates = [m.win_rate for m in self.metrics_history]
                plt.figure(figsize=(12, 6))
                plt.plot(win_rates, marker="o")
                plt.title("Win Rate Over Time")
                plt.ylabel("Win Rate (%)")
                plt.xlabel("Evaluation Period")
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / "win_rate_trend.png")
                plt.close()

                # Profit factor
                pf = [m.profit_factor for m in self.metrics_history]
                plt.figure(figsize=(12, 6))
                plt.plot(pf, marker="s", color="green")
                plt.axhline(y=1.5, color="r", linestyle="--", label="Good Threshold (1.5)")
                plt.title("Profit Factor Over Time")
                plt.ylabel("Profit Factor")
                plt.xlabel("Evaluation Period")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / "profit_factor_trend.png")
                plt.close()

            logger.info(f"Performance plots saved to {output_dir}")
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


class ContinuousEvaluator:
    """Continuous evaluation system for production monitoring."""

    def __init__(self, evaluation_interval_trades: int = 50):
        """Initialize continuous evaluator."""
        self.evaluation_interval_trades = evaluation_interval_trades
        self.trade_buffer: List[Dict] = []
        self.evaluation_history: List[Dict] = []

    def add_trade(self, trade: Dict) -> None:
        """Add trade to buffer."""
        self.trade_buffer.append(trade)

    def should_evaluate(self) -> bool:
        """Check if evaluation threshold reached."""
        return len(self.trade_buffer) >= self.evaluation_interval_trades

    def evaluate(self, pipeline: EvaluationPipeline) -> Optional[TradeMetrics]:
        """Evaluate if threshold reached."""
        if not self.should_evaluate():
            return None

        metrics = pipeline.evaluate_trading_performance(self.trade_buffer)
        self.evaluation_history.append(metrics.to_dict())
        self.trade_buffer = []

        return metrics

    def detect_performance_degradation(self, threshold: float = 0.05) -> bool:
        """
        Detect if performance has degraded.

        Args:
            threshold: Minimum acceptable win rate change

        Returns:
            True if degradation detected
        """
        if len(self.evaluation_history) < 2:
            return False

        current_wr = self.evaluation_history[-1].get("win_rate", 0)
        previous_wr = self.evaluation_history[-2].get("win_rate", 0)

        degradation = previous_wr - current_wr
        return degradation > threshold

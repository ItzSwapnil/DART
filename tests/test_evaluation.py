"""Tests for the evaluation and monitoring pipeline."""

import pytest

from ml.evaluation_monitor import ContinuousEvaluator, EvaluationPipeline


@pytest.fixture
def eval_pipeline() -> EvaluationPipeline:
    """Fixture to initialize EvaluationPipeline."""
    return EvaluationPipeline(project_name="Test-Project", use_wandb=False)


def test_evaluate_trading_performance_empty(eval_pipeline: EvaluationPipeline) -> None:
    """Test performance evaluation with no trades."""
    metrics = eval_pipeline.evaluate_trading_performance([])
    assert metrics.total_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.profit_factor == 0.0


def test_evaluate_trading_performance_metrics(eval_pipeline: EvaluationPipeline) -> None:
    """Test standard trading performance metrics calculation."""
    # 5 wins of $20, 5 losses of $10
    trades = [
        {"profit": 20.0},
        {"profit": 20.0},
        {"profit": 20.0},
        {"profit": 20.0},
        {"profit": 20.0},
        {"profit": -10.0},
        {"profit": -10.0},
        {"profit": -10.0},
        {"profit": -10.0},
        {"profit": -10.0},
    ]

    metrics = eval_pipeline.evaluate_trading_performance(trades, initial_balance=1000.0)

    assert metrics.total_trades == 10
    assert metrics.winning_trades == 5
    assert metrics.losing_trades == 5
    assert metrics.win_rate == 0.5
    assert metrics.total_profit == 100.0
    assert metrics.total_loss == 50.0
    assert metrics.profit_factor == 2.0
    assert metrics.avg_profit_per_trade == 20.0
    assert metrics.avg_loss_per_trade == 10.0


def test_continuous_evaluator() -> None:
    """Test ContinuousEvaluator trade buffering and performance degradation checks."""
    evaluator = ContinuousEvaluator(evaluation_interval_trades=3)
    pipeline = EvaluationPipeline(use_wandb=False)

    # Initial state
    assert evaluator.should_evaluate() is False

    evaluator.add_trade({"profit": 10.0})
    evaluator.add_trade({"profit": -5.0})
    assert evaluator.should_evaluate() is False

    evaluator.add_trade({"profit": 15.0})
    assert evaluator.should_evaluate() is True

    # Run evaluation
    metrics = evaluator.evaluate(pipeline)
    assert metrics is not None
    assert metrics.total_trades == 3
    # Buffer should be reset
    assert len(evaluator.trade_buffer) == 0
    assert len(evaluator.evaluation_history) == 1


def test_detect_performance_degradation() -> None:
    """Test performance degradation warnings work correctly."""
    evaluator = ContinuousEvaluator(evaluation_interval_trades=3)

    # Setup history manually
    evaluator.evaluation_history = [
        {"win_rate": 0.70},
        {"win_rate": 0.60},  # Degraded by 10%
    ]

    # 10% degradation should exceed 5% threshold
    assert evaluator.detect_performance_degradation(threshold=0.05) is True
    # 10% degradation should NOT exceed 15% threshold
    assert evaluator.detect_performance_degradation(threshold=0.15) is False

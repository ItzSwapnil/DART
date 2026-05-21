"""
DART v3.0 - State-of-the-Art Deep Adaptive Reinforcement Trader
Main entry point with modern architecture and SOTA AI components.
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dart_main")

# Import SOTA components
from config.dart_config import get_config, load_config
from ml.evaluation_monitor import EvaluationPipeline
from ml.trading_ai_v3 import TradingAIv3
from ml.llm_market_analyzer import LLMMarketAnalyzer
from api.deriv_client import DerivClient


async def initialize_dart() -> dict:
    """Initialize DART v3.0 components."""
    logger.info("=" * 80)
    logger.info("DART v3.0 - State-of-the-Art AI Trading System")
    logger.info("=" * 80)

    # Load configuration
    config = get_config()
    logger.info(f"\n{config}\n")

    # Validate configuration
    if not config.validate_configuration():
        logger.warning("Configuration validation warnings detected")

    # Initialize components
    components = {
        "config": config,
        "eval_pipeline": None,
        "trading_ai": None,
        "llm_analyzer": None,
        "deriv_client": None,
    }

    # Initialize evaluation pipeline
    try:
        components["eval_pipeline"] = EvaluationPipeline(
            project_name=config.app_name,
            use_wandb=config.monitoring.wandb_enabled,
        )
        logger.info("✓ Evaluation pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize evaluation pipeline: {e}")

    # Initialize LLM Market Analyzer
    if config.ai.use_llm_analysis:
        try:
            components["llm_analyzer"] = LLMMarketAnalyzer(
                model_name=config.ai.llm_model,
                base_url=config.api.llm_base_url,
                allow_fallback=not config.ai.require_real_llm,
            )
            logger.info("✓ LLM Market Analyzer initialized")
        except Exception as e:
            if config.ai.require_real_llm:
                raise RuntimeError(
                    "Real LLM is required but unavailable. "
                    "Start vLLM and ensure the configured model is installed."
                ) from e
            logger.warning(f"LLM Market Analyzer not available: {e}")

    # Initialize Trading AI v3.0
    try:
        components["trading_ai"] = TradingAIv3(
            model_dir=config.model_dir,
            use_llm=config.ai.use_llm_analysis,
            require_real_llm=config.ai.require_real_llm,
            eval_pipeline=components["eval_pipeline"],
        )
        logger.info("✓ Trading AI v3.0 initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Trading AI: {e}")

    # Initialize Deriv Client
    if config.api.deriv_access_token:
        try:
            components["deriv_client"] = DerivClient(
                app_id=config.api.deriv_app_id,
                access_token=config.api.deriv_access_token,
                account_id=config.api.deriv_account_id or None,
            )
            logger.info("✓ Deriv API client initialized (OAuth 2.0)")
        except Exception as e:
            logger.warning(f"Deriv client initialization: {e}")
    else:
        logger.info("Deriv OAuth token not configured — trading disabled")

    logger.info("\n" + "=" * 80)
    logger.info("DART v3.0 initialized successfully!")
    logger.info("=" * 80 + "\n")

    return components


async def demo_trading_workflow(components: dict) -> None:
    """Demonstrate trading workflow."""
    logger.info("Demo: Trading Workflow")
    logger.info("-" * 80)

    config = components["config"]
    trading_ai = components["trading_ai"]
    llm_analyzer = components["llm_analyzer"]

    # Create sample market data
    import numpy as np
    import pandas as pd

    logger.info("\n1. Generating sample market data...")
    market_data = []
    price = 100.0
    for i in range(100):
        price = price * (1 + np.random.normal(0.0001, 0.01))
        market_data.append({
            "open": price * 0.99,
            "high": price * 1.01,
            "low": price * 0.98,
            "close": price,
            "volume": np.random.randint(1000, 10000),
        })

    logger.info(f"✓ Generated {len(market_data)} candles")

    # Train model
    logger.info("\n2. Training Trading AI...")
    train_metrics = trading_ai.train(market_data)
    if train_metrics:
        logger.info(f"✓ Training completed: {train_metrics}")

    # Generate strategy
    logger.info("\n3. Generating trading strategy...")
    strategy = trading_ai.generate_strategy(market_data, use_llm=True)
    if strategy:
        logger.info(f"✓ Strategy: {strategy['direction']}")
        logger.info(f"  Confidence: {strategy['confidence']:.2%}")
        if "llm_analysis" in strategy:
            logger.info(f"  LLM Analysis: {strategy['llm_analysis'].get('trend', 'N/A')}")

    # Demonstrate evaluation
    logger.info("\n4. Demonstrating evaluation pipeline...")
    if components["eval_pipeline"]:
        sample_trades = [
            {"profit": 10.0},
            {"profit": -5.0},
            {"profit": 15.0},
            {"profit": -3.0},
            {"profit": 20.0},
        ]
        metrics = components["eval_pipeline"].evaluate_trading_performance(sample_trades)
        logger.info(f"✓ Evaluation Results:")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    logger.info("\n" + "-" * 80)
    logger.info("Demo completed!\n")


async def main():
    """Main entry point."""
    try:
        # Initialize all components
        components = await initialize_dart()

        # Run demo
        await demo_trading_workflow(components)

        # Show next steps
        logger.info("=" * 80)
        logger.info("NEXT STEPS FOR PRODUCTION:")
        logger.info("-" * 80)
        logger.info("1. Configure .env with OAuth 2.0 credentials (DERIV_ACCESS_TOKEN, DERIV_ACCOUNT_ID)")
        logger.info("2. Install vLLM for LLM analysis: https://vllm.ai")
        logger.info("3. Run vLLM OpenAI-compatible server: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct")
        logger.info("4. Set up Weights & Biases for monitoring")
        logger.info("5. Run training: await components['trading_ai'].train(data)")
        logger.info("6. Enable auto-trading in config")
        logger.info("7. Monitor with evaluation pipeline")
        logger.info("=" * 80 + "\n")

        logger.info("To start the dashboard UI: python ui/app.py")
        logger.info("To run auto-trader: python ml/auto_trader.py\n")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

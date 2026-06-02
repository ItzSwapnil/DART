"""
DART v3.0 - State-of-the-Art Deep Adaptive Reinforcement Trader
Main entry point with modern architecture and SOTA AI components.
"""

import asyncio
import logging
from typing import Any

from utils.logging import configure_logging

# Configure centralized logging first
configure_logging(level="INFO")
logger = logging.getLogger("dart_main")

# Import SOTA components
from api.deriv_client import DerivClient
from config.dart_config import get_config
from ml.evaluation_monitor import EvaluationPipeline
from ml.llm_market_analyzer import LLMMarketAnalyzer
from ml.trading_ai_v3 import TradingAIv3


async def initialize_dart() -> dict[str, Any]:
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
    components: dict[str, Any] = {
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
                    "Real LLM is required but unavailable. Start vLLM and ensure the configured model is installed."
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
    """Demonstrate trading workflow with real market data and real virtual-account trade execution."""
    logger.info("Demo: Real Trading Workflow")
    logger.info("-" * 80)

    config = components["config"]
    trading_ai = components["trading_ai"]
    client = components.get("deriv_client")

    if not client:
        logger.error("Deriv client not initialized. Please ensure API__DERIV_ACCESS_TOKEN is set in your .env file.")
        return

    # Check connection and get account details
    logger.info("Connecting to Deriv API and retrieving account information...")
    account_info = await client.get_account_info()
    if not account_info:
        logger.error(
            "Failed to connect or fetch account information. Please check your credentials and internet connection."
        )
        return

    balance_info = await client.get_account_balance()
    balance = balance_info.get("balance", 0.0) if balance_info else 0.0
    currency = balance_info.get("currency", "USD") if balance_info else "USD"
    logger.info(f"✓ Connected to Deriv. Account ID: {client.account_id} | Balance: {balance} {currency}")

    # Fetch real market data
    symbol = config.trading.symbol or "R_75"
    granularity = config.trading.granularity or 60
    logger.info(f"\n1. Fetching real historical candles for {symbol} (granularity={granularity}s)...")
    market_data = await client.get_historical_data(symbol=symbol, granularity=granularity, days=1)

    if not market_data or len(market_data) < 20:
        logger.error(
            f"Failed to retrieve sufficient historical data for {symbol} (received {len(market_data) if market_data else 0} candles)"
        )
        return

    logger.info(f"✓ Retrieved {len(market_data)} real candles")

    # Train model
    logger.info("\n2. Training Trading AI on real market data...")
    train_metrics = trading_ai.train(market_data)
    if train_metrics:
        logger.info(f"✓ Training completed: {train_metrics}")

    # Generate strategy
    logger.info("\n3. Generating trading strategy...")
    strategy = trading_ai.generate_strategy(market_data, use_llm=config.ai.use_llm_analysis)
    if not strategy:
        logger.error("Failed to generate strategy decision.")
        return

    logger.info(f"✓ AI Strategy Decision: {strategy['direction']}")
    logger.info(f"  Confidence: {strategy['confidence']:.2%}")
    if "llm_analysis" in strategy:
        logger.info(f"  LLM Analysis: {strategy['llm_analysis'].get('trend', 'N/A')}")

    # Execute trade
    logger.info("\n4. Placing live demo trade on virtual account...")
    amount = config.trading.base_trade_amount or 10.0
    duration = strategy.get("duration", 60)
    duration_unit = strategy.get("duration_unit", "s")

    logger.info(f"Buying {strategy['direction']} contract for {amount} USD (duration={duration}{duration_unit})...")
    trade_resp = await client.buy_contract(
        symbol=symbol,
        contract_type=strategy["direction"],
        amount=amount,
        duration=duration,
        duration_unit=duration_unit,
    )

    if "error" in trade_resp:
        logger.error(f"Trade execution failed: {trade_resp['error'].get('message')}")
        return

    contract_id = trade_resp.get("buy", {}).get("contract_id")
    if not contract_id:
        logger.error("Trade response missing contract_id")
        return

    logger.info(f"✓ Trade successfully executed! Contract ID: {contract_id}")

    # Poll trade status
    logger.info("\n5. Monitoring trade progress...")
    final_info = None
    while True:
        status_info = await client.check_trade_status(str(contract_id))
        if not status_info:
            await asyncio.sleep(2)
            continue

        status = status_info.get("status")
        current_spot = status_info.get("current_spot")
        logger.info(f"  [Monitoring] Status: {status} | Spot: {current_spot}")

        if status in ["sold", "expired", "won", "lost"]:
            final_info = status_info
            break

        await asyncio.sleep(3)

    buy_price = float(final_info.get("buy_price", 0) or 0)
    sell_price = float(final_info.get("sell_price", 0) or 0)
    profit = sell_price - buy_price
    logger.info(f"\n✓ Trade Completed! Final Status: {final_info.get('status')}")
    logger.info(f"  Invested: ${buy_price:.2f} | Return: ${sell_price:.2f} | Profit/Loss: ${profit:.2f}")

    # Demonstrate evaluation
    logger.info("\n6. Running performance evaluation...")
    if components["eval_pipeline"]:
        from datetime import datetime

        trade_result = {
            "contract_id": contract_id,
            "symbol": symbol,
            "profit": profit,
            "status": final_info.get("status"),
            "contract_info": final_info,
            "timestamp": datetime.now().isoformat(),
        }
        # Update AI model metrics
        trading_ai.update_from_trade_results([trade_result])

        metrics = components["eval_pipeline"].evaluate_trading_performance([trade_result])
        logger.info("✓ Session Evaluation:")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    logger.info("\n" + "-" * 80)
    logger.info("Demo trading workflow completed successfully!\n")


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
        logger.info(
            "3. Run vLLM OpenAI-compatible server: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct"
        )
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

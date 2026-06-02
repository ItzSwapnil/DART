"""
DART v3.0 - Modern Web Dashboard (Dash Edition)
A stunning, professional web-based interface featuring glassmorphism design,
real-time AI insights, and advanced trading analytics powered by Dash.
"""

# Force CPU mode for PyTorch is commented out to allow GPU usage as requested by user
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import asyncio
import logging
import os
import sys
import threading
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html

# Prioritize local DART imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.deriv_client import DerivClient
from config.dart_config import get_config
from ml.auto_trader import AutoTrader
from ml.trading_ai_v3 import TradingAIv3

logger = logging.getLogger("trading_dashboard")

# Load Configuration
config = get_config()
CONFIDENCE_THRESHOLD = config.risk.confidence_threshold
USE_DEEP_RL = config.ai.use_deep_rl
USE_ENHANCED_FEATURES = config.ai.use_ensemble_ml
MAX_DAILY_LOSS = config.risk.max_daily_loss
TRADE_AMOUNT = config.trading.base_trade_amount
DERIV_APP_ID = config.api.deriv_app_id
DERIV_ACCESS_TOKEN = config.api.deriv_access_token
DERIV_ACCOUNT_ID = config.api.deriv_account_id

# ---------------------------------------------------------------------------
# Background Logging Handler
# ---------------------------------------------------------------------------


class ListLogHandler(logging.Handler):
    """Custom logging handler that collects log records in thread-safe lists."""

    def __init__(self):
        super().__init__()
        self.logs = []
        self.train_logs = []
        self.analyze_logs = []
        self.logs_lock = threading.RLock()

    def emit(self, record):
        try:
            msg = self.format(record)
            with self.logs_lock:
                item = {
                    "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                    "level": record.levelname,
                    "message": msg,
                }
                self.logs.append(item)
                if len(self.logs) > 100:
                    self.logs.pop(0)

                # Route to training logs
                if any(k in msg.lower() for k in ["train", "model", "candles", "collecting", "historical data"]):
                    self.train_logs.append(item)
                    if len(self.train_logs) > 100:
                        self.train_logs.pop(0)

                # Route to analysis logs
                if any(
                    k in msg.lower()
                    for k in ["analyz", "strategy", "signal", "direction", "sentiment", "confidence", "regime"]
                ):
                    self.analyze_logs.append(item)
                    if len(self.analyze_logs) > 100:
                        self.analyze_logs.pop(0)
        except Exception:
            self.handleError(record)

    def emit_custom(self, level, message):
        """Manually append a log message to the log records list."""
        with self.logs_lock:
            item = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": logging.getLevelName(level),
                "message": message,
            }
            self.logs.append(item)
            if len(self.logs) > 100:
                self.logs.pop(0)

            # Route to training logs
            if any(k in message.lower() for k in ["train", "model", "candles", "collecting", "historical data"]):
                self.train_logs.append(item)
                if len(self.train_logs) > 100:
                    self.train_logs.pop(0)

            # Route to analysis logs
            if any(
                k in message.lower()
                for k in ["analyz", "strategy", "signal", "direction", "sentiment", "confidence", "regime"]
            ):
                self.analyze_logs.append(item)
                if len(self.analyze_logs) > 100:
                    self.analyze_logs.pop(0)

    def get_logs(self):
        with self.logs_lock:
            return list(self.logs)

    def get_train_logs(self):
        with self.logs_lock:
            return list(self.train_logs)

    def get_analyze_logs(self):
        with self.logs_lock:
            return list(self.analyze_logs)


log_handler = ListLogHandler()
# Suppress deep noise and add handler to root logger
logging.getLogger().addHandler(log_handler)
for noisy in ("urllib3", "httpx", "httpcore", "websockets", "asyncio", "werkzeug"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Backend & Global Instance Lifecycle
# ---------------------------------------------------------------------------

client = DerivClient(
    app_id=DERIV_APP_ID,
    access_token=DERIV_ACCESS_TOKEN or None,
    account_id=DERIV_ACCOUNT_ID or None,
)

trading_ai = TradingAIv3(
    model_dir=config.model_dir,
    use_llm=config.ai.use_llm_analysis,
    require_real_llm=False,
)

auto_trader = AutoTrader(client=client, trading_ai=trading_ai)


def status_callback(status_data):
    msg = str(status_data.get("message") or "")
    status = str(status_data.get("status") or "INFO").upper()

    # Update active task state and progress dynamically
    if "collecting historical data" in msg.lower() or "collecting" in msg.lower():
        DASHBOARD_STATE["active_tasks"]["training"] = "Deriv API: Collecting historical candle data..."
        DASHBOARD_STATE["active_tasks"]["training_progress"] = 40
    elif "training model on" in msg.lower():
        DASHBOARD_STATE["active_tasks"]["training"] = msg
        DASHBOARD_STATE["active_tasks"]["training_progress"] = 75
    elif "model training completed" in msg.lower() or "trained" in status.lower():
        DASHBOARD_STATE["active_tasks"]["training"] = "Model trained successfully!"
        DASHBOARD_STATE["active_tasks"]["training_progress"] = 100

    if "strategy generated" in msg.lower() or "strategy" in status.lower():
        DASHBOARD_STATE["active_tasks"]["analysis"] = msg
        DASHBOARD_STATE["active_tasks"]["analysis_progress"] = 90
    elif "analyzing" in msg.lower():
        DASHBOARD_STATE["active_tasks"]["analysis"] = msg
        DASHBOARD_STATE["active_tasks"]["analysis_progress"] = 45

    if status == "ERROR":
        log_handler.emit_custom(logging.ERROR, msg)
    elif status == "WARNING":
        log_handler.emit_custom(logging.WARNING, msg)
    elif status in ["SUCCESS", "EXECUTED", "COMPLETED"]:
        log_handler.emit_custom(logging.INFO, f"SUCCESS: {msg}")
    else:
        log_handler.emit_custom(logging.INFO, msg)


auto_trader.register_status_callback(status_callback)

# ---------------------------------------------------------------------------
# Async Background Loop Thread
# ---------------------------------------------------------------------------

bg_loop = None


def start_background_loop():
    global bg_loop
    bg_loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(bg_loop)
        bg_loop.run_forever()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()


def run_coro(coro):
    """Run a coroutine on the background thread's loop and return the result synchronously."""
    return asyncio.run_coroutine_threadsafe(coro, bg_loop).result()


# ---------------------------------------------------------------------------
# State Synchronization Loop
# ---------------------------------------------------------------------------

DASHBOARD_STATE = {
    "connected": False,
    "balance": "$0.00 USD",
    "win_rate": 0.0,
    "active_trades_count": 0,
    "active_symbols": "None",
    "ai_confidence": "N/A",
    "ai_status": "No Signal",
    "session_pnl": 0.0,
    "active_trades": [],
    "recent_trades": [],
    "logs": [],
    "sentiment_score": 0.5,
    "uncertainty": 0.15,
    "regime_name": "Normal",
    "available_markets": [
        "Synthetics - Volatility 75 Index",
        "Forex - EUR/USD",
        "Forex - GBP/USD",
        "Forex - USD/JPY",
        "Forex - AUD/USD",
    ],
    "symbols_dict": {
        "Synthetics - Volatility 75 Index": "R_75",
        "Forex - EUR/USD": "frxEURUSD",
        "Forex - GBP/USD": "frxGBPUSD",
        "Forex - USD/JPY": "frxUSDJPY",
        "Forex - AUD/USD": "frxAUDUSD",
        "Volatility 75 Index": "R_75",
        "EUR/USD": "frxEURUSD",
        "GBP/USD": "frxGBPUSD",
        "USD/JPY": "frxUSDJPY",
        "AUD/USD": "frxAUDUSD",
    },
    "trading_active": False,
    "active_tasks": {
        "training": None,
        "training_progress": 0,
        "analysis": None,
        "analysis_progress": 0,
    },
    "train_logs": [],
    "analyze_logs": [],
    "selected_market": "Synthetics - Volatility 75 Index",
    "selected_timeframe": "1 minute",
    "candles_df": pd.DataFrame(),
    "market_categories": {
        "Synthetics": ["Synthetics - Volatility 75 Index"],
        "Derived": ["Synthetics - Volatility 75 Index"],
        "Forex": [
            "Forex - EUR/USD",
            "Forex - GBP/USD",
            "Forex - USD/JPY",
            "Forex - AUD/USD",
        ],
    },
}

symbols_loaded = False


async def update_dashboard_state_loop():
    """Poll Deriv API and background objects, maintaining thread-safe dashboard variables."""
    global symbols_loaded
    while True:
        try:
            connected = await client.check_connection()
            DASHBOARD_STATE["connected"] = connected

            if connected:
                if not symbols_loaded:
                    try:
                        symbols_dict = await client.get_active_symbols()
                        if symbols_dict:
                            categories = {
                                "Derived": [],
                                "Baskets": [],
                                "Synthetics": [],
                                "Forex": [],
                                "Stock Indices": [],
                                "Cryptocurrencies": [],
                                "Commodities": [],
                            }

                            symbols_map = {}
                            display_names = []

                            for item in client.active_symbols_raw:
                                symbol = item.get("symbol") or item.get("underlying_symbol")
                                display_name = item.get("display_name") or item.get("underlying_symbol_name") or symbol
                                raw_market = str(item.get("market") or "").lower()
                                raw_submarket = str(item.get("submarket") or "").lower()

                                # Determine the specific categories this symbol belongs to
                                symbol_categories = []
                                
                                if "basket" in raw_submarket or "basket" in raw_market:
                                    category_display = "Baskets"
                                    symbol_categories.append("Baskets")
                                    symbol_categories.append("Derived")
                                elif raw_market == "synthetic_index":
                                    category_display = "Synthetics"
                                    symbol_categories.append("Synthetics")
                                    symbol_categories.append("Derived")
                                elif raw_market == "forex":
                                    category_display = "Forex"
                                    symbol_categories.append("Forex")
                                elif raw_market == "indices":
                                    category_display = "Stock Indices"
                                    symbol_categories.append("Stock Indices")
                                elif raw_market == "cryptocurrency":
                                    category_display = "Cryptocurrencies"
                                    symbol_categories.append("Cryptocurrencies")
                                elif raw_market == "commodities" or "metal" in raw_submarket:
                                    category_display = "Commodities"
                                    symbol_categories.append("Commodities")
                                else:
                                    category_display = "Derived"
                                    symbol_categories.append("Derived")

                                name = f"{category_display} - {display_name}"
                                if not item.get("exchange_is_open", True):
                                    name = f"{name} [CLOSED]"

                                display_names.append(name)
                                for cat in symbol_categories:
                                    categories[cat].append(name)

                                # Map all representations of name to symbol in symbols_dict
                                symbols_map[symbol] = symbol
                                symbols_map[display_name] = symbol
                                for cat in ["Derived", "Baskets", "Synthetics", "Forex", "Stock Indices", "Cryptocurrencies", "Commodities"]:
                                    symbols_map[f"{cat} - {display_name}"] = symbol
                                    symbols_map[f"{cat} - {display_name} [CLOSED]"] = symbol

                            DASHBOARD_STATE["symbols_dict"] = symbols_map
                            DASHBOARD_STATE["available_markets"] = sorted(list(set(display_names)))
                            DASHBOARD_STATE["market_categories"] = categories
                            symbols_loaded = True
                            logger.info(f"Successfully loaded and categorized {len(display_names)} unique symbols from Deriv API.")
                    except Exception as sym_err:
                        logger.error(f"Error loading symbols list: {sym_err}")

                # Update Account Balance
                bal_info = await client.get_account_balance()
                if bal_info:
                    DASHBOARD_STATE["balance"] = f"${float(bal_info['balance']):,.2f} {bal_info['currency']}"

                # Check active positions and trade history
                await client.check_all_trades()

                # Fetch candles for selected market
                selected_market = DASHBOARD_STATE.get("selected_market", "Synthetics - Volatility 75 Index")
                selected_timeframe = DASHBOARD_STATE.get("selected_timeframe", "1 minute")
                symbols_map = DASHBOARD_STATE["symbols_dict"]
                raw_symbol = symbols_map.get(selected_market, selected_market)
                granularity = TIMEFRAME_MAPPING.get(selected_timeframe, 60)

                try:
                    candles = await client.get_candles(raw_symbol, granularity, count=100)
                    if candles:
                        df = pd.DataFrame(candles)
                        if "epoch" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["epoch"], unit="s")
                        elif "time" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["time"], unit="s")

                        for col in ["open", "high", "low", "close"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col])

                        if "volume" not in df.columns:
                            df["volume"] = 0
                        else:
                            df["volume"] = pd.to_numeric(df["volume"])

                        DASHBOARD_STATE["candles_df"] = df
                except Exception as candle_err:
                    logger.error(f"Error background loading candles for {selected_market}: {candle_err}")

            # Update metrics from trade history
            trades = client.trade_history
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if float(t.get("profit") or 0.0) > 0)
            DASHBOARD_STATE["win_rate"] = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            DASHBOARD_STATE["session_pnl"] = sum(float(t.get("profit") or 0.0) for t in trades)
            DASHBOARD_STATE["recent_trades"] = list(trades[-5:])

            # Active positions
            DASHBOARD_STATE["active_trades"] = list(client.active_trades.values())
            DASHBOARD_STATE["active_trades_count"] = len(client.active_trades)
            DASHBOARD_STATE["active_symbols"] = (
                ", ".join({t.get("symbol", "") for t in client.active_trades.values()}) or "None"
            )

            # Auto-trader status
            DASHBOARD_STATE["trading_active"] = auto_trader.is_running

            # AI Insight Status
            selected_market = DASHBOARD_STATE.get("selected_market", "Synthetics - Volatility 75 Index")
            symbols_map = DASHBOARD_STATE["symbols_dict"]
            raw_symbol = symbols_map.get(selected_market, selected_market)
            active_model = auto_trader.get_model_for_symbol(raw_symbol)
            last_strat = active_model.last_strategy if active_model else None
            if last_strat:
                unc_dict = last_strat.get("uncertainty", {})
                unc_val = 0.15
                if isinstance(unc_dict, dict):
                    unc_val = unc_dict.get("total", unc_dict.get("uncertainty", 0.15))
                elif isinstance(unc_dict, (int, float)):
                    unc_val = unc_dict
                DASHBOARD_STATE["sentiment_score"] = last_strat.get("probability", 0.5)
                DASHBOARD_STATE["uncertainty"] = unc_val
                DASHBOARD_STATE["regime_name"] = last_strat.get("market_regime", "Normal")
                DASHBOARD_STATE["ai_confidence"] = f"{last_strat.get('confidence', 0.5) * 100:.1f}%"
                DASHBOARD_STATE["ai_status"] = last_strat.get("direction", "HOLD")

            # Get latest logs
            DASHBOARD_STATE["logs"] = log_handler.get_logs()
            DASHBOARD_STATE["train_logs"] = log_handler.get_train_logs()
            DASHBOARD_STATE["analyze_logs"] = log_handler.get_analyze_logs()

        except Exception as e:
            logger.error(f"Error updating dashboard state: {e}")

        await asyncio.sleep(1.5)


# Start background sync
start_background_loop()
asyncio.run_coroutine_threadsafe(update_dashboard_state_loop(), bg_loop)

# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

TIMEFRAME_MAPPING = {
    "1 minute": 60,
    "5 minutes": 300,
    "15 minutes": 900,
    "1 hour": 3600,
    "4 hours": 14400,
    "1 day": 86400,
}


def get_market_data(market, timeframe="1 minute", count=100):
    """Retrieve historical candle dataframe from the background-thread cache.

    Never performs a blocking network call — the background sync loop
    populates ``DASHBOARD_STATE["candles_df"]`` every 1.5 s, so callers
    always get the latest cached snapshot instantly.
    """
    df = DASHBOARD_STATE.get("candles_df")
    if df is not None and not df.empty:
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotly Graph Builders
# ---------------------------------------------------------------------------


def make_price_chart_figure(market, timeframe):
    df = get_market_data(market, timeframe, 100)

    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            template="plotly_dark",
            title=f"No data available for {market} (Connecting to API...)",
            height=450,
            xaxis={"visible": False},
            yaxis={"visible": False},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=market,
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        )
    )

    # Volume Trace (using secondary axis)
    if "volume" in df.columns and not (df["volume"] == 0).all():
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.15,
                marker_color="#3b82f6",
            )
        )

    # Moving Averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["sma_20"],
            mode="lines",
            name="SMA 20",
            line={"color": "#f59e0b", "width": 1.5},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["sma_50"],
            mode="lines",
            name="SMA 50",
            line={"color": "#8b5cf6", "width": 1.5},
        )
    )

    fig.update_layout(
        yaxis={"title": "Price", "side": "left", "showgrid": True, "gridcolor": "rgba(255, 255, 255, 0.05)"},
        yaxis2={"title": "Volume", "side": "right", "overlaying": "y", "showgrid": False},
        xaxis={"title": "Time", "showgrid": True, "gridcolor": "rgba(255, 255, 255, 0.05)"},
        template="plotly_dark",
        height=450,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 50, "b": 50, "l": 50, "r": 50},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def make_sentiment_gauge(score, threshold_val=0.55):
    threshold_scaled = float(threshold_val) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Market Sentiment", "font": {"size": 15, "family": "Outfit"}},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, 25], "color": "#ef4444"},
                    {"range": [25, 50], "color": "#f59e0b"},
                    {"range": [50, 75], "color": "#10b981"},
                    {"range": [75, 100], "color": "#06b6d4"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": threshold_scaled,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=220,
        margin={"t": 40, "b": 10, "l": 30, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_equity_curve_figure():
    trades = client.trade_history
    if not trades:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title="No trades executed yet",
            height=220,
            xaxis={"visible": False},
            yaxis={"visible": False},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    try:
        bal = float(DASHBOARD_STATE["balance"].split()[0].replace("$", "").replace(",", ""))
    except Exception:
        bal = 10000.0

    balance_history = [bal]
    for t in reversed(trades):
        profit = float(t.get("profit") or 0.0)
        prev_balance = balance_history[-1] - profit
        balance_history.append(prev_balance)
    balance_history.reverse()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(balance_history))),
            y=balance_history,
            mode="lines+markers",
            name="Portfolio Value",
            line={"color": "#10b981", "width": 2.5},
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.05)",
        )
    )

    fig.update_layout(
        margin={"t": 20, "b": 40, "l": 40, "r": 20},
        xaxis_title="Trades",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Dash Application Initialization
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "description", "content": "DART v3.0 - Deep Adaptive Reinforcement Trader dashboard"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
    title="DART v3.0 - Web Dashboard",
)

# Custom premium layout HTML structure
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #090a16 0%, #12132a 100%);
                color: #f1f5f9;
                font-family: 'Inter', sans-serif;
                min-height: 100vh;
            }
            .glass-card {
                background: rgba(22, 24, 47, 0.55);
                border: 1px solid rgba(56, 189, 248, 0.1);
                border-radius: 16px;
                backdrop-filter: blur(12px);
                padding: 24px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                margin-bottom: 20px;
            }
            .glass-card:hover {
                border-color: rgba(56, 189, 248, 0.25);
                box-shadow: 0 12px 40px rgba(56, 189, 248, 0.15);
                transform: translateY(-2px);
            }
            .metric-card {
                background: linear-gradient(135deg, rgba(30, 32, 66, 0.6) 0%, rgba(15, 17, 36, 0.6) 100%);
                border: 1px solid rgba(56, 189, 248, 0.15);
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-4px);
                border-color: rgba(56, 189, 248, 0.3);
                box-shadow: 0 12px 30px rgba(56, 189, 248, 0.2);
            }
            .glow-text {
                background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: 'Outfit', sans-serif;
            }
            .sidebar {
                background: rgba(10, 11, 23, 0.85);
                backdrop-filter: blur(20px);
                border-right: 1px solid rgba(255, 255, 255, 0.05);
                padding: 30px 20px;
                height: 100vh;
                overflow-y: auto;
                position: sticky;
                top: 0;
            }
            .console-log {
                background-color: #070814;
                border: 1px solid rgba(56, 189, 248, 0.1);
                border-radius: 12px;
                font-family: 'Courier New', Courier, monospace;
                font-size: 0.85rem;
                padding: 16px;
                height: 280px;
                overflow-y: auto;
                color: #a7f3d0;
            }
            .log-item {
                margin-bottom: 6px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.02);
                padding-bottom: 4px;
                word-break: break-all;
            }
            .trading-card {
                background: rgba(30, 32, 66, 0.4);
                border: 1px solid rgba(56, 189, 248, 0.1);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                transition: all 0.2s ease;
            }
            .trading-card:hover {
                background: rgba(30, 32, 66, 0.6);
                border-color: rgba(56, 189, 248, 0.2);
            }
            .form-control, .form-select {
                background-color: rgba(30, 32, 66, 0.6) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: #f1f5f9 !important;
                border-radius: 8px !important;
            }
            .form-control:focus, .form-select:focus {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25) !important;
                background-color: rgba(30, 32, 66, 0.8) !important;
            }
            /* Style Dash Dropdown (Radix select button and menu popup options) */
            .dash-dropdown {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid rgba(0, 0, 0, 0.2) !important;
            }
            .dash-dropdown,
            .dash-dropdown * {
                color: #000000 !important;
            }
            .dash-dropdown-content,
            .dash-dropdown-content * {
                color: #000000 !important;
            }
            .dash-dropdown-option:hover,
            .dash-dropdown-option.selected {
                background-color: #e2e8f0 !important;
            }
            .btn-primary-glow {
                background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: 600;
                padding: 10px 20px;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
                transition: all 0.3s ease;
            }
            .btn-primary-glow:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
                background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
            }
            .btn-stop-glow {
                background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: 600;
                padding: 10px 20px;
                box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
                transition: all 0.3s ease;
            }
            .btn-stop-glow:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(239, 68, 68, 0.5);
            }
            .btn-secondary-glow {
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                color: #e2e8f0;
                font-weight: 600;
                padding: 10px 20px;
                transition: all 0.3s ease;
                width: 100%;
            }
            .btn-secondary-glow:hover {
                background: rgba(30, 41, 59, 0.9);
                border-color: rgba(255, 255, 255, 0.2);
                color: white;
            }
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #090a16;
            }
            ::-webkit-scrollbar-thumb {
                background: #1e293b;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #334155;
            }
            /* Style Dash Slider (rc-slider) components */
            .rc-slider-rail {
                background-color: rgba(255, 255, 255, 0.1) !important;
            }
            .rc-slider-track {
                background-color: #3b82f6 !important;
            }
            .rc-slider-handle {
                border: 2px solid #3b82f6 !important;
                background-color: #1e293b !important;
            }
            .rc-slider-handle:hover, .rc-slider-handle:active, .rc-slider-handle-click-focused:focus {
                border-color: #8b5cf6 !important;
            }
            .rc-slider-mark-text {
                color: #94a3b8 !important;
            }
            .rc-slider-mark-text-active {
                color: #f1f5f9 !important;
            }
            .rc-slider-tooltip-inner {
                background-color: #1e293b !important;
                color: #f1f5f9 !important;
                border: 1px solid rgba(56, 189, 248, 0.2) !important;
            }
            /* Style any slider input text boxes */
            .dash-range-slider-input, .rc-slider-tooltip {
                background-color: rgba(30, 32, 66, 0.6) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: #f1f5f9 !important;
                border-radius: 4px !important;
                padding: 2px 6px !important;
                text-align: center !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ---------------------------------------------------------------------------
# Application Layout
# ---------------------------------------------------------------------------

app.layout = dbc.Container(
    [
        # High-frequency polling component
        dcc.Interval(id="interval-component", interval=1500, n_intervals=0),
        dbc.Row(
            [
                # Sidebar Configuration Panel
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H3("🎯 DART v3.0", className="glow-text text-center font-weight-bold mb-2"),
                                html.P(
                                    "Deep Adaptive Reinforcement Trader",
                                    className="text-muted text-center mb-4Small",
                                    style={"fontSize": "0.9rem"},
                                ),
                                # Connection badge
                                html.Div(id="connection-status-badge", className="mb-4 text-center"),
                                html.Hr(style={"borderColor": "rgba(255, 255, 255, 0.1)"}),
                                html.H5("📈 Market Settings", className="mb-3 text-info"),
                                html.Label("Select Category", className="mb-2"),
                                dcc.Dropdown(
                                    id="category-selector",
                                    options=[
                                        {"label": "All Markets", "value": "All"},
                                        {"label": "Derived", "value": "Derived"},
                                        {"label": "Baskets", "value": "Baskets"},
                                        {"label": "Synthetics", "value": "Synthetics"},
                                        {"label": "Forex", "value": "Forex"},
                                        {"label": "Stock Indices", "value": "Stock Indices"},
                                        {"label": "Cryptocurrencies", "value": "Cryptocurrencies"},
                                        {"label": "Commodities", "value": "Commodities"},
                                    ],
                                    value="All",
                                    clearable=False,
                                    className="mb-3 dash-dropdown",
                                ),
                                html.Label("Select Markets", className="mb-2"),
                                dcc.Dropdown(
                                    id="market-selector", options=[], multi=True, className="mb-3 dash-dropdown"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "✅ Select All",
                                                id="select-all-category-btn",
                                                className="btn-secondary-glow mb-3",
                                                style={"fontSize": "0.8rem", "padding": "6px"},
                                            ),
                                            width=6,
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                "❌ Clear All",
                                                id="clear-all-selected-btn",
                                                className="btn-secondary-glow mb-3",
                                                style={"fontSize": "0.8rem", "padding": "6px"},
                                            ),
                                            width=6,
                                        ),
                                    ],
                                    className="g-2",
                                ),
                                html.Label("Timeframe", className="mb-2"),
                                dcc.Dropdown(
                                    id="timeframe-selector",
                                    options=[{"label": t, "value": t} for t in TIMEFRAME_MAPPING],
                                    value="1 minute",
                                    clearable=False,
                                    className="mb-4 dash-dropdown",
                                ),
                                html.Hr(style={"borderColor": "rgba(255, 255, 255, 0.1)"}),
                                html.H5("🤖 AI Configuration", className="mb-3 text-info"),
                                html.Label("Confidence Threshold", className="mb-2"),
                                dcc.Slider(
                                    id="confidence-slider",
                                    min=0.1,
                                    max=0.9,
                                    step=0.05,
                                    value=CONFIDENCE_THRESHOLD,
                                    marks={i / 10: str(i / 10) for i in range(1, 10)},
                                    className="mb-4",
                                ),
                                dbc.Checklist(
                                    id="ai-switches",
                                    options=[
                                        {"label": "Enable Deep RL", "value": "deep_rl"},
                                        {"label": "Enhanced Features", "value": "enhanced"},
                                    ],
                                    value=["deep_rl", "enhanced"],
                                    switch=True,
                                    className="mb-4",
                                ),
                                html.Hr(style={"borderColor": "rgba(255, 255, 255, 0.1)"}),
                                html.H5("⚠️ Risk Management", className="mb-3 text-info"),
                                html.Label("Max Daily Loss ($)", className="mb-2"),
                                dbc.Input(
                                    id="max-daily-loss-input",
                                    type="number",
                                    value=MAX_DAILY_LOSS,
                                    min=10,
                                    className="mb-3",
                                ),
                                html.Label("Trade Amount ($)", className="mb-2"),
                                dbc.Input(
                                    id="trade-amount-input", type="number", value=TRADE_AMOUNT, min=1, className="mb-4"
                                ),
                                html.Hr(style={"borderColor": "rgba(255, 255, 255, 0.1)"}),
                                html.H5("🎮 Controls", className="mb-3 text-info"),
                                # Train Model Section with Mini Progress Bar
                                html.Div(
                                    [
                                        dbc.Progress(
                                            id="train-mini-progress",
                                            value=0,
                                            striped=True,
                                            animated=True,
                                            color="warning",
                                            style={
                                                "height": "6px",
                                                "marginBottom": "6px",
                                                "borderRadius": "3px",
                                                "display": "none",
                                            },
                                        ),
                                        dbc.Button(
                                            "🎯 Train Model", id="train-btn", className="btn-secondary-glow mb-2 w-100"
                                        ),
                                    ],
                                    className="mb-1",
                                ),
                                # Analyze Market Section with Mini Progress Bar
                                html.Div(
                                    [
                                        dbc.Progress(
                                            id="analyze-mini-progress",
                                            value=0,
                                            striped=True,
                                            animated=True,
                                            color="info",
                                            style={
                                                "height": "6px",
                                                "marginBottom": "6px",
                                                "borderRadius": "3px",
                                                "display": "none",
                                            },
                                        ),
                                        dbc.Button(
                                            "📊 Analyze Market",
                                            id="analyze-btn",
                                            className="btn-secondary-glow mb-4 w-100",
                                        ),
                                    ],
                                    className="mb-1",
                                ),
                                html.Div(
                                    dbc.Button(
                                        "🚀 Start Auto Trading",
                                        id="trading-btn",
                                        className="btn-primary-glow w-100",
                                    ),
                                    id="trading-button-container",
                                    className="mb-2",
                                ),
                                html.Div(id="task-status-container", className="mt-2"),
                            ],
                            className="sidebar",
                        )
                    ],
                    width=3,
                    style={"padding": 0},
                ),
                # Main Dashboard View
                dbc.Col(
                    [
                        # Main header row
                        html.Div(
                            [
                                html.H1("🎯 DART Main Dashboard", className="glow-text mt-4 mb-2"),
                                html.P(
                                    "Professional Real-Time Autonomous AI Trading Terminal", className="text-muted mb-4"
                                ),
                            ],
                            style={"padding": "0 20px"},
                        ),
                        # Metrics Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H5(
                                                "💰 Account Balance",
                                                className="text-muted mb-2",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.H3(
                                                id="balance-val",
                                                style={
                                                    "color": "#10b981",
                                                    "fontWeight": "bold",
                                                    "fontSize": "1.35rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                        ],
                                        className="metric-card",
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H5(
                                                "📈 Win Rate",
                                                className="text-muted mb-2",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.H3(
                                                id="win-rate-val",
                                                style={
                                                    "color": "#3b82f6",
                                                    "fontWeight": "bold",
                                                    "fontSize": "1.35rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.P(
                                                id="win-rate-sub",
                                                className="text-muted mb-0",
                                                style={
                                                    "fontSize": "0.75rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                        ],
                                        className="metric-card",
                                    ),
                                    width=2,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H5(
                                                "💼 Active Trades",
                                                className="text-muted mb-2",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.H3(
                                                id="active-trades-val",
                                                style={
                                                    "color": "#f59e0b",
                                                    "fontWeight": "bold",
                                                    "fontSize": "1.35rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.P(
                                                id="active-trades-sub",
                                                className="text-muted mb-0",
                                                style={
                                                    "fontSize": "0.75rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                        ],
                                        className="metric-card",
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H5(
                                                "🤖 AI Confidence",
                                                className="text-muted mb-2",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.H3(
                                                id="ai-confidence-val",
                                                style={
                                                    "color": "#8b5cf6",
                                                    "fontWeight": "bold",
                                                    "fontSize": "1.35rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.P(
                                                id="ai-confidence-sub",
                                                className="text-muted mb-0",
                                                style={
                                                    "fontSize": "0.75rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                        ],
                                        className="metric-card",
                                    ),
                                    width=2,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H5(
                                                "⚡ Session P&L",
                                                className="text-muted mb-2",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                            html.H3(
                                                id="session-pnl-val",
                                                style={
                                                    "fontWeight": "bold",
                                                    "fontSize": "1.35rem",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                },
                                            ),
                                        ],
                                        className="metric-card",
                                    ),
                                    width=2,
                                ),
                            ],
                            className="mb-4",
                            style={"padding": "0 20px"},
                        ),
                        # Charts and AI Insights
                        dbc.Row(
                            [
                                # Main Candle Chart
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4(id="chart-title", className="text-light font-weight-bold mb-3"),
                                            dcc.Graph(id="price-chart", config={"displayModeBar": False}),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=8,
                                ),
                                # AI Insight Module
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4("🤖 AI Insights", className="text-light font-weight-bold mb-3"),
                                            dcc.Graph(id="sentiment-gauge", config={"displayModeBar": False}),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Strong(
                                                                    "🎯 Confidence",
                                                                    style={"fontSize": "0.8rem", "color": "#94a3b8"},
                                                                ),
                                                                html.Div(
                                                                    id="ai-conf-val",
                                                                    style={
                                                                        "fontSize": "1.3rem",
                                                                        "color": "#10b981",
                                                                        "fontWeight": "bold",
                                                                    },
                                                                ),
                                                            ],
                                                            className="trading-card",
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Strong(
                                                                    "📊 Regime",
                                                                    style={"fontSize": "0.8rem", "color": "#94a3b8"},
                                                                ),
                                                                html.Div(
                                                                    id="ai-regime-val",
                                                                    style={
                                                                        "fontSize": "1.1rem",
                                                                        "color": "#8b5cf6",
                                                                        "fontWeight": "bold",
                                                                    },
                                                                ),
                                                            ],
                                                            className="trading-card",
                                                        ),
                                                        width=6,
                                                    ),
                                                ]
                                            ),
                                            html.H5("Current Signal Status", className="text-muted mb-2"),
                                            html.Div(id="current-signal-div"),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=4,
                                ),
                            ],
                            className="mb-4",
                            style={"padding": "0 20px"},
                        ),
                        # Performance stats + Active Positions
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4(
                                                "💼 Active Positions", className="text-light font-weight-bold mb-3"
                                            ),
                                            html.Div(id="active-positions-container"),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4(
                                                "📈 Equity Performance (Session)",
                                                className="text-light font-weight-bold mb-3",
                                            ),
                                            dcc.Graph(id="equity-chart", config={"displayModeBar": False}),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=6,
                                ),
                            ],
                            className="mb-4",
                            style={"padding": "0 20px"},
                        ),
                        # Training and Analysis Console logs
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4(
                                                "⚙️ Model Training Console", className="text-light font-weight-bold mb-3"
                                            ),
                                            html.Div(
                                                id="train-logs-container",
                                                className="console-log",
                                                style={"color": "#fcd34d", "height": "200px"},
                                            ),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4(
                                                "🔍 Market Analysis Console",
                                                className="text-light font-weight-bold mb-3",
                                            ),
                                            html.Div(
                                                id="analyze-logs-container",
                                                className="console-log",
                                                style={"color": "#67e8f9", "height": "200px"},
                                            ),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=6,
                                ),
                            ],
                            className="mb-4",
                            style={"padding": "0 20px"},
                        ),
                        # Recent activity and Logs
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4("🔄 Recent Trades", className="text-light font-weight-bold mb-3"),
                                            html.Div(id="recent-trades-container"),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H4(
                                                "📝 System Console Logs", className="text-light font-weight-bold mb-3"
                                            ),
                                            html.Div(id="logs-container", className="console-log"),
                                        ],
                                        className="glass-card",
                                    ),
                                    width=6,
                                ),
                            ],
                            style={"padding": "0 20px", "marginBottom": "50px"},
                        ),
                    ],
                    width=9,
                    style={"padding": 0},
                ),
            ]
        ),
    ],
    fluid=True,
    style={"padding": 0},
)

# ---------------------------------------------------------------------------
# Callback Handlers
# ---------------------------------------------------------------------------


@app.callback(
    [Output("market-selector", "options"), Output("market-selector", "value")],
    [
        Input("category-selector", "value"),
        Input("select-all-category-btn", "n_clicks"),
        Input("clear-all-selected-btn", "n_clicks"),
        Input("interval-component", "n_intervals"),
    ],
    State("market-selector", "value"),
)
def update_market_dropdown(category, select_all_clicks, clear_all_clicks, n_intervals, current_value):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Get available markets for selected category
    categories_map = DASHBOARD_STATE.get("market_categories", {})

    if category == "All" or not categories_map:
        category_markets = list(DASHBOARD_STATE["available_markets"])
    else:
        category_markets = list(categories_map.get(category, []))

    # Initialize current value safely
    val = current_value or []

    # Handle clear all action
    if triggered_id == "clear-all-selected-btn" and clear_all_clicks:
        val = []
    # Handle select all action
    elif triggered_id == "select-all-category-btn" and select_all_clicks:
        # Add all category markets to the selected value (union)
        val = list(set(val + category_markets))

    # Ensure options contains all category markets, PLUS any currently selected markets
    # so that selected markets from other categories do not disappear from the UI!
    all_options_set = set(category_markets + val)
    options_list = sorted(list(all_options_set))
    options = [{"label": m, "value": m} for m in options_list]

    # Ensure default value if empty
    if not val and category_markets:
        val = [category_markets[0]]

    return options, val


@app.callback(
    [
        Output("connection-status-badge", "children"),
        Output("balance-val", "children"),
        Output("win-rate-val", "children"),
        Output("win-rate-sub", "children"),
        Output("active-trades-val", "children"),
        Output("active-trades-sub", "children"),
        Output("ai-confidence-val", "children"),
        Output("ai-confidence-sub", "children"),
        Output("session-pnl-val", "children"),
        Output("session-pnl-val", "style"),
        Output("ai-conf-val", "children"),
        Output("ai-regime-val", "children"),
        Output("sentiment-gauge", "figure"),
        Output("current-signal-div", "children"),
        Output("active-positions-container", "children"),
        Output("recent-trades-container", "children"),
        Output("logs-container", "children"),
        Output("trading-button-container", "children"),
        Output("equity-chart", "figure"),
        Output("train-btn", "disabled"),
        Output("train-btn", "children"),
        Output("analyze-btn", "disabled"),
        Output("analyze-btn", "children"),
        Output("task-status-container", "children"),
        Output("train-mini-progress", "value"),
        Output("train-mini-progress", "style"),
        Output("analyze-mini-progress", "value"),
        Output("analyze-mini-progress", "style"),
        Output("train-logs-container", "children"),
        Output("analyze-logs-container", "children"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("train-btn", "n_clicks"),
        Input("analyze-btn", "n_clicks"),
        Input("trading-btn", "n_clicks"),
    ],
    [
        State("market-selector", "value"),
        State("timeframe-selector", "value"),
        State("confidence-slider", "value"),
        State("ai-switches", "value"),
        State("max-daily-loss-input", "value"),
        State("trade-amount-input", "value"),
    ],
)
def update_dashboard_metrics(
    n_intervals,
    n_train,
    n_analyze,
    n_trading,
    markets,
    timeframe,
    confidence_threshold,
    ai_switches,
    max_daily_loss,
    trade_amount,
):
    try:
        return _update_dashboard_metrics_impl(
            n_intervals,
            n_train,
            n_analyze,
            n_trading,
            markets,
            timeframe,
            confidence_threshold,
            ai_switches,
            max_daily_loss,
            trade_amount,
        )
    except Exception as e:
        import traceback

        err_msg = f"CALLBACK ERROR: {e}\n{traceback.format_exc()}"
        logger.error(err_msg)
        print(err_msg, flush=True)
        raise e


def _update_dashboard_metrics_impl(
    n_intervals,
    n_train,
    n_analyze,
    n_trading,
    markets,
    timeframe,
    confidence_threshold,
    ai_switches,
    max_daily_loss,
    trade_amount,
):
    """Update all dashboard counters, lists, charts, and logs from sync variables."""
    # Dynamically update the running auto-trader's parameters with the user's slider/input adjustments
    if auto_trader.is_running:
        auto_trader.custom_confidence_threshold = confidence_threshold
        auto_trader.base_trade_amount = trade_amount
        config.risk.confidence_threshold = confidence_threshold
        config.trading.base_trade_amount = trade_amount
        config.risk.max_daily_loss = max_daily_loss
        config.ai.use_deep_rl = "deep_rl" in ai_switches
        config.ai.use_ensemble_ml = "enhanced" in ai_switches

    ctx = callback_context
    triggered_id = None
    if ctx.triggered:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Identify active markets
    selected_market = "Synthetics - Volatility 75 Index"
    if markets and len(markets) > 0:
        selected_market = markets[0]

    DASHBOARD_STATE["selected_market"] = selected_market
    DASHBOARD_STATE["selected_timeframe"] = timeframe

    symbols_map = DASHBOARD_STATE["symbols_dict"]
    raw_symbol = symbols_map.get(selected_market, selected_market)
    granularity = TIMEFRAME_MAPPING.get(timeframe, 60)

    # Build the full list of raw symbols for all selected markets
    raw_symbols_list = [symbols_map.get(m, m) for m in (markets or [selected_market])]

    # Perform action if triggered by button click
    if triggered_id == "train-btn" and n_train:
        # Check if already training to prevent duplicate threads
        if not DASHBOARD_STATE["active_tasks"]["training"]:
            # Launch model training thread for ALL selected markets
            DASHBOARD_STATE["active_tasks"]["training"] = f"Initializing training for {len(raw_symbols_list)} markets..."
            DASHBOARD_STATE["active_tasks"]["training_progress"] = 5
            t = threading.Thread(target=run_background_train, args=(raw_symbols_list, granularity), daemon=True)
            t.start()

    elif triggered_id == "analyze-btn" and n_analyze:
        # Check if already analyzing to prevent duplicate threads
        if not DASHBOARD_STATE["active_tasks"]["analysis"]:
            # Launch market analysis thread for ALL selected markets
            DASHBOARD_STATE["active_tasks"]["analysis"] = f"Initializing analysis for {len(raw_symbols_list)} markets..."
            DASHBOARD_STATE["active_tasks"]["analysis_progress"] = 5
            t = threading.Thread(target=run_background_analyze, args=(raw_symbols_list, granularity), daemon=True)
            t.start()

    elif triggered_id == "trading-btn" and n_trading:
        if not DASHBOARD_STATE["trading_active"]:
            log_handler.emit_custom(logging.INFO, "Starting automated trading portfolio loop...")
            raw_symbols_list = [symbols_map.get(m, m) for m in (markets or [selected_market])]
            config.risk.confidence_threshold = confidence_threshold
            config.risk.max_daily_loss = max_daily_loss
            config.trading.base_trade_amount = trade_amount
            config.ai.use_deep_rl = "deep_rl" in ai_switches
            config.ai.use_ensemble_ml = "enhanced" in ai_switches
            success = auto_trader.start_trading(
                symbol=raw_symbols_list,
                granularity=granularity,
                confidence_threshold=confidence_threshold,
                loop=bg_loop,
            )
            if success:
                DASHBOARD_STATE["trading_active"] = True
                log_handler.emit_custom(logging.INFO, "Automated trading loop successfully launched!")
            else:
                log_handler.emit_custom(logging.ERROR, "Failed to start automated trading loop.")
        else:
            log_handler.emit_custom(logging.INFO, "Halting automated trading loop...")
            success = auto_trader.stop_trading()
            if success:
                DASHBOARD_STATE["trading_active"] = False
                log_handler.emit_custom(logging.INFO, "Automated trading loop stopped.")
            else:
                log_handler.emit_custom(logging.WARNING, "Trading loop was not running.")

    # 1. Connection Status Badge
    is_connected = DASHBOARD_STATE["connected"]
    status_text = "🟢 Connected" if is_connected else "🔴 Disconnected"
    status_color = "success" if is_connected else "danger"
    status_badge = dbc.Badge(status_text, color=status_color, style={"fontSize": "1rem", "padding": "6px 12px"})

    # 2. Account Balance
    balance_text = DASHBOARD_STATE["balance"]

    # 3. Win Rate
    win_rate = DASHBOARD_STATE["win_rate"]
    total_trades = len(client.trade_history)
    wins = sum(1 for t in client.trade_history if float(t.get("profit") or 0.0) > 0)
    win_rate_sub = f"{wins} wins out of {total_trades} trades"

    # 4. Active Trades count & list
    active_count = DASHBOARD_STATE["active_trades_count"]
    active_symbols = DASHBOARD_STATE["active_symbols"]

    # 5. AI Confidence
    ai_confidence = DASHBOARD_STATE["ai_confidence"]
    ai_status = DASHBOARD_STATE["ai_status"]
    ai_sub = f"Signal: {ai_status}"

    # 6. Session PNL
    pnl = DASHBOARD_STATE["session_pnl"]
    pnl_sign = "+" if pnl >= 0 else ""
    pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
    pnl_text = f"{pnl_sign}${pnl:,.2f}"
    pnl_style = {
        "color": pnl_color,
        "fontWeight": "bold",
        "fontSize": "1.35rem",
        "whiteSpace": "nowrap",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    }

    # 7. AI Panels
    sentiment_score = DASHBOARD_STATE["sentiment_score"]
    uncertainty = DASHBOARD_STATE["uncertainty"]
    regime = DASHBOARD_STATE["regime_name"]
    ai_conf_text = DASHBOARD_STATE["ai_confidence"]

    sentiment_gauge = make_sentiment_gauge(sentiment_score, confidence_threshold)

    # Signal badge
    signal_color = "#10b981" if ai_status == "CALL" else "#ef4444" if ai_status == "PUT" else "#f59e0b"
    if ai_status != "No Signal":
        signal_badge = html.Div(
            [
                html.Span(ai_status, style={"color": signal_color, "fontWeight": "bold", "fontSize": "1.3rem"}),
                html.Span("Just Now", style={"float": "right", "color": "#94a3b8"}),
                html.Br(),
                html.Small(
                    f"Confidence: {ai_confidence} | Uncertainty: {uncertainty * 100:.1f}%", style={"color": "#94a3b8"}
                ),
            ],
            className="trading-card",
            style={"borderLeft": f"4px solid {signal_color}"},
        )
    else:
        signal_badge = html.Div(
            "No active signals. Click 'Analyze Market' to generate a trade decision.", className="text-muted"
        )

    # 8. Active Positions List
    positions_list = []
    if not DASHBOARD_STATE["active_trades"]:
        positions_list = html.Div("No active positions currently.", className="text-muted")
    else:
        for pos in DASHBOARD_STATE["active_trades"]:
            curr_info = pos.get("current_info", {})
            buy_price = float(curr_info.get("buy_price") or pos.get("amount", 0.0))
            current_spot = curr_info.get("current_spot")
            entry_spot = curr_info.get("entry_spot")

            pnl_val = 0.0
            pnl_text_pos = "Pending"
            pnl_color_pos = "#6b7280"

            if current_spot and entry_spot:
                is_call = pos["contract_type"] == "CALL"
                won = (current_spot > entry_spot) if is_call else (current_spot < entry_spot)
                if won:
                    pnl_val = float(curr_info.get("payout", buy_price * 1.8) or (buy_price * 1.8)) - buy_price
                    pnl_text_pos = f"+${pnl_val:.2f}"
                    pnl_color_pos = "#10b981"
                else:
                    pnl_val = -buy_price
                    pnl_text_pos = f"-${buy_price:.2f}"
                    pnl_color_pos = "#ef4444"

            positions_list.append(
                html.Div(
                    [
                        html.Strong(pos["symbol"]),
                        html.Span(f" - {pos['contract_type']} (ID: {pos.get('id', 'N/A')})"),
                        html.Span(pnl_text_pos, style={"float": "right", "color": pnl_color_pos, "fontWeight": "bold"}),
                        html.Br(),
                        html.Small(f"Stake: ${buy_price:.2f} | Duration: {pos['duration']}{pos['duration_unit']}"),
                    ],
                    className="trading-card",
                )
            )

    # 9. Recent Completed Trades List
    recent_list = []
    if not DASHBOARD_STATE["recent_trades"]:
        recent_list = html.Div("No completed trades recorded in this session.", className="text-muted")
    else:
        for trade in reversed(DASHBOARD_STATE["recent_trades"]):
            trade_pnl = float(trade.get("profit") or 0.0)
            trade_color = "#10b981" if trade_pnl >= 0 else "#ef4444"
            trade_pnl_str = f"+${trade_pnl:.2f}" if trade_pnl >= 0 else f"-${abs(trade_pnl):.2f}"
            status_str = "WIN" if trade_pnl >= 0 else "LOSS"

            start_time = trade.get("start_time")
            time_str = start_time.strftime("%H:%M:%S") if isinstance(start_time, datetime) else "Completed"

            recent_list.append(
                html.Div(
                    [
                        html.Small(time_str, style={"color": "#94a3b8"}),
                        html.Span(trade_pnl_str, style={"float": "right", "color": trade_color, "fontWeight": "bold"}),
                        html.Br(),
                        html.Strong(trade.get("symbol")),
                        html.Span(f" {trade.get('contract_type')} - "),
                        html.Span(status_str, style={"color": trade_color, "fontWeight": "bold"}),
                    ],
                    className="trading-card",
                )
            )

    # 10. Console Logs
    log_elements = []
    for record in reversed(DASHBOARD_STATE["logs"]):
        level_color = "#3b82f6"
        if record["level"] in ["WARNING", "WARN"]:
            level_color = "#f59e0b"
        elif record["level"] in ["ERROR", "CRITICAL"]:
            level_color = "#ef4444"

        log_elements.append(
            html.Div(
                [
                    html.Span(f"[{record['time']}] ", style={"color": "#64748b"}),
                    html.Span(f"{record['level']} ", style={"color": level_color, "fontWeight": "bold"}),
                    html.Span(record["message"]),
                ],
                className="log-item",
            )
        )

    # 11. Trading action button toggle
    trading_active = DASHBOARD_STATE["trading_active"]
    if trading_active:
        btn_action = dbc.Button("🛑 Stop Auto Trading", id="trading-btn", className="btn-stop-glow w-100")
    else:
        btn_action = dbc.Button("🚀 Start Auto Trading", id="trading-btn", className="btn-primary-glow w-100")

    # 12. Equity curves
    equity_curve = make_equity_curve_figure()

    # 13. Dynamic button states & active tasks info
    active_tasks = DASHBOARD_STATE.get(
        "active_tasks", {"training": None, "training_progress": 0, "analysis": None, "analysis_progress": 0}
    )

    train_disabled = False
    train_children = "🎯 Train Model"
    if active_tasks.get("training"):
        train_disabled = True
        train_children = html.Span([dbc.Spinner(size="sm", spinnerClassName="me-2"), "Training..."])

    analyze_disabled = False
    analyze_children = "📊 Analyze Market"
    if active_tasks.get("analysis"):
        analyze_disabled = True
        analyze_children = html.Span([dbc.Spinner(size="sm", spinnerClassName="me-2"), "Analyzing..."])

    task_status_panel = []
    if active_tasks.get("training") or active_tasks.get("analysis"):
        status_items = []
        if active_tasks.get("training"):
            prog_train = active_tasks.get("training_progress", 0)
            status_items.append(
                html.Div(
                    [
                        html.Small(
                            "⚙️ MODEL TRAINING:", style={"color": "#f59e0b", "fontWeight": "bold", "fontSize": "0.75rem"}
                        ),
                        html.P(
                            active_tasks["training"],
                            style={"color": "#f1f5f9", "fontSize": "0.85rem", "margin": "0 0 6px 0"},
                        ),
                        dbc.Progress(
                            value=prog_train,
                            striped=True,
                            animated=True,
                            color="warning",
                            label=f"{prog_train}%" if prog_train > 10 else "",
                            style={
                                "height": "16px",
                                "borderRadius": "8px",
                                "fontSize": "0.75rem",
                                "fontWeight": "bold",
                            },
                        ),
                    ],
                    className="mb-3",
                )
            )
        if active_tasks.get("analysis"):
            prog_analyze = active_tasks.get("analysis_progress", 0)
            status_items.append(
                html.Div(
                    [
                        html.Small(
                            "🔍 MARKET ANALYSIS:",
                            style={"color": "#06b6d4", "fontWeight": "bold", "fontSize": "0.75rem"},
                        ),
                        html.P(
                            active_tasks["analysis"],
                            style={"color": "#f1f5f9", "fontSize": "0.85rem", "margin": "0 0 6px 0"},
                        ),
                        dbc.Progress(
                            value=prog_analyze,
                            striped=True,
                            animated=True,
                            color="info",
                            label=f"{prog_analyze}%" if prog_analyze > 10 else "",
                            style={
                                "height": "16px",
                                "borderRadius": "8px",
                                "fontSize": "0.75rem",
                                "fontWeight": "bold",
                            },
                        ),
                    ]
                )
            )
        task_status_panel = html.Div(
            status_items,
            className="glass-card mb-3",
            style={
                "padding": "12px 16px",
                "background": "rgba(245, 158, 11, 0.1)" if active_tasks.get("training") else "rgba(6, 182, 212, 0.1)",
                "border": "1px solid rgba(245, 158, 11, 0.25)"
                if active_tasks.get("training")
                else "1px solid rgba(6, 182, 212, 0.25)",
                "borderRadius": "12px",
                "boxShadow": "0 4px 15px rgba(0,0,0,0.2)",
            },
        )

    # Calculate mini progress bar values and styles
    prog_train = active_tasks.get("training_progress", 0)
    if active_tasks.get("training"):
        train_mini_style = {"height": "6px", "marginBottom": "6px", "borderRadius": "3px"}
        train_mini_val = prog_train
    else:
        train_mini_style = {"display": "none"}
        train_mini_val = 0

    prog_analyze = active_tasks.get("analysis_progress", 0)
    if active_tasks.get("analysis"):
        analyze_mini_style = {"height": "6px", "marginBottom": "6px", "borderRadius": "3px"}
        analyze_mini_val = prog_analyze
    else:
        analyze_mini_style = {"display": "none"}
        analyze_mini_val = 0

    # Training logs
    train_log_elements = []
    if not DASHBOARD_STATE["train_logs"]:
        train_log_elements = html.Div("No training logs in this session.", className="text-muted")
    else:
        for record in reversed(DASHBOARD_STATE["train_logs"]):
            level_color = (
                "#f59e0b"
                if record["level"] in ["WARNING", "WARN"]
                else "#ef4444"
                if record["level"] in ["ERROR", "CRITICAL"]
                else "#fcd34d"
            )
            train_log_elements.append(
                html.Div(
                    [
                        html.Span(f"[{record['time']}] ", style={"color": "#64748b"}),
                        html.Span(f"{record['level']} ", style={"color": level_color, "fontWeight": "bold"}),
                        html.Span(record["message"]),
                    ],
                    className="log-item",
                )
            )

    # Analysis logs
    analyze_log_elements = []
    if not DASHBOARD_STATE["analyze_logs"]:
        analyze_log_elements = html.Div("No analysis logs in this session.", className="text-muted")
    else:
        for record in reversed(DASHBOARD_STATE["analyze_logs"]):
            level_color = (
                "#f59e0b"
                if record["level"] in ["WARNING", "WARN"]
                else "#ef4444"
                if record["level"] in ["ERROR", "CRITICAL"]
                else "#67e8f9"
            )
            analyze_log_elements.append(
                html.Div(
                    [
                        html.Span(f"[{record['time']}] ", style={"color": "#64748b"}),
                        html.Span(f"{record['level']} ", style={"color": level_color, "fontWeight": "bold"}),
                        html.Span(record["message"]),
                    ],
                    className="log-item",
                )
            )

    return (
        status_badge,
        balance_text,
        f"{win_rate:.1f}%",
        win_rate_sub,
        str(active_count),
        active_symbols,
        ai_confidence,
        ai_sub,
        pnl_text,
        pnl_style,
        ai_conf_text,
        regime,
        sentiment_gauge,
        signal_badge,
        positions_list,
        recent_list,
        log_elements,
        btn_action,
        equity_curve,
        train_disabled,
        train_children,
        analyze_disabled,
        analyze_children,
        task_status_panel,
        train_mini_val,
        train_mini_style,
        analyze_mini_val,
        analyze_mini_style,
        train_log_elements,
        analyze_log_elements,
    )


@app.callback(
    [Output("price-chart", "figure"), Output("chart-title", "children")],
    [
        Input("market-selector", "value"),
        Input("timeframe-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_chart(markets, timeframe, n):
    """Update primary market candlestick chart upon dropdown change or refresh interval."""
    # Find active market (first selected item or fallback index)
    selected_market = DASHBOARD_STATE.get("selected_market", "Synthetics - Volatility 75 Index")
    if markets and len(markets) > 0:
        selected_market = markets[0]

    # State is authoritative from update_dashboard_metrics; only read here.
    fig = make_price_chart_figure(selected_market, timeframe)
    title = f"📈 {selected_market} Price Stream - {timeframe}"
    return fig, title


# ---------------------------------------------------------------------------
# Background Async Actions Thread Pools
# ---------------------------------------------------------------------------


def run_background_train(raw_symbols, granularity):
    """Call model training asynchronously for ALL selected markets."""
    try:
        import time

        if isinstance(raw_symbols, str):
            raw_symbols = [raw_symbols]

        total = len(raw_symbols)
        succeeded = 0
        failed = 0

        log_handler.emit_custom(logging.INFO, f"Starting batch model training for {total} market(s)...")

        for idx, symbol in enumerate(raw_symbols, 1):
            progress = int((idx - 1) / total * 100)
            DASHBOARD_STATE["active_tasks"]["training"] = f"Training {idx}/{total}: {symbol}..."
            DASHBOARD_STATE["active_tasks"]["training_progress"] = max(progress, 5)
            log_handler.emit_custom(logging.INFO, f"[{idx}/{total}] Initiating model training for {symbol}...")

            try:
                success = run_coro(auto_trader.train_model(symbol, granularity))
                if success:
                    succeeded += 1
                    log_handler.emit_custom(logging.INFO, f"[{idx}/{total}] SUCCESS: Model trained for {symbol}!")
                else:
                    failed += 1
                    log_handler.emit_custom(logging.ERROR, f"[{idx}/{total}] FAILED: Model training failed for {symbol}.")
            except Exception as sym_err:
                failed += 1
                log_handler.emit_custom(logging.ERROR, f"[{idx}/{total}] ERROR training {symbol}: {sym_err}")

        DASHBOARD_STATE["active_tasks"]["training"] = f"Batch complete: {succeeded} trained, {failed} failed"
        DASHBOARD_STATE["active_tasks"]["training_progress"] = 100
        log_handler.emit_custom(
            logging.INFO,
            f"Batch training complete! {succeeded}/{total} succeeded, {failed}/{total} failed.",
        )
        time.sleep(2.0)

    except Exception as e:
        log_handler.emit_custom(logging.ERROR, f"Background batch training exception: {e}")
    finally:
        DASHBOARD_STATE["active_tasks"]["training"] = None
        DASHBOARD_STATE["active_tasks"]["training_progress"] = 0


def run_background_analyze(raw_symbols, granularity):
    """Call market analysis asynchronously for ALL selected markets."""
    try:
        import time

        if isinstance(raw_symbols, str):
            raw_symbols = [raw_symbols]

        total = len(raw_symbols)
        results = []

        log_handler.emit_custom(logging.INFO, f"Starting batch AI analysis for {total} market(s)...")

        for idx, symbol in enumerate(raw_symbols, 1):
            progress = int((idx - 1) / total * 100)
            DASHBOARD_STATE["active_tasks"]["analysis"] = f"Analyzing {idx}/{total}: {symbol}..."
            DASHBOARD_STATE["active_tasks"]["analysis_progress"] = max(progress, 5)

            try:
                # Auto-train if model has not been trained yet for this symbol
                has_been_trained = symbol in auto_trader.symbol_last_training
                if not has_been_trained:
                    DASHBOARD_STATE["active_tasks"]["training"] = f"Auto-training {symbol} before analysis..."
                    DASHBOARD_STATE["active_tasks"]["training_progress"] = 25
                    log_handler.emit_custom(logging.WARNING, f"[{idx}/{total}] No trained model for {symbol}. Auto-training...")
                    train_ok = run_coro(auto_trader.train_model(symbol, granularity))
                    DASHBOARD_STATE["active_tasks"]["training"] = None
                    DASHBOARD_STATE["active_tasks"]["training_progress"] = 0
                    if not train_ok:
                        log_handler.emit_custom(logging.ERROR, f"[{idx}/{total}] Auto-training failed for {symbol}. Skipping.")
                        results.append((symbol, "SKIP", 0))
                        continue

                log_handler.emit_custom(logging.INFO, f"[{idx}/{total}] Generating AI strategy for {symbol}...")
                strategy = run_coro(auto_trader.generate_strategy(symbol, granularity))
                if strategy:
                    direction = strategy.get("direction", "HOLD")
                    confidence = strategy.get("confidence", 0.0) * 100
                    results.append((symbol, direction, confidence))
                    log_handler.emit_custom(
                        logging.INFO, f"[{idx}/{total}] {symbol}: {direction} (Confidence: {confidence:.1f}%)"
                    )
                else:
                    results.append((symbol, "FAIL", 0))
                    log_handler.emit_custom(logging.ERROR, f"[{idx}/{total}] Strategy generation failed for {symbol}.")
            except Exception as sym_err:
                results.append((symbol, "ERROR", 0))
                log_handler.emit_custom(logging.ERROR, f"[{idx}/{total}] ERROR analyzing {symbol}: {sym_err}")

        # Summary
        calls = sum(1 for _, d, _ in results if d == "CALL")
        puts = sum(1 for _, d, _ in results if d == "PUT")
        holds = sum(1 for _, d, _ in results if d == "HOLD")
        errors = sum(1 for _, d, _ in results if d in ("FAIL", "ERROR", "SKIP"))
        DASHBOARD_STATE["active_tasks"]["analysis"] = (
            f"Batch complete: {calls} CALL, {puts} PUT, {holds} HOLD, {errors} errors"
        )
        DASHBOARD_STATE["active_tasks"]["analysis_progress"] = 100
        log_handler.emit_custom(
            logging.INFO,
            f"Batch analysis complete! {calls} CALL, {puts} PUT, {holds} HOLD, {errors} errors out of {total} markets.",
        )
        time.sleep(2.0)

    except Exception as e:
        log_handler.emit_custom(logging.ERROR, f"Background batch analysis exception: {e}")
    finally:
        DASHBOARD_STATE["active_tasks"]["analysis"] = None
        DASHBOARD_STATE["active_tasks"]["analysis_progress"] = 0


# Interaction controls are now handled directly inside update_dashboard_metrics callback for immediate responsiveness.


# ---------------------------------------------------------------------------
# Main Runner Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

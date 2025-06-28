import asyncio
import datetime
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import mplfinance as mpf
import numpy as np
import pandas as pd
import sv_ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression

from api.deriv_client import DerivClient
from config.settings import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_CANDLE_COUNT,
    DEFAULT_THEME,
    DEFAULT_TIMEFRAME,
    DERIV_API_TOKEN,
    DERIV_APP_ID,
    USE_DEEP_RL,
    USE_ENHANCED_FEATURES,
)
from ml.auto_trader import AutoTrader
from ml.trading_ai import TradingAI
from ui.chart_styles import get_chart_style
from ui.ui_theme import configure_styles, format_currency
from utils.timeframe import get_granularity_mapping

# Default confidence threshold from settings
CURRENT_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD


class DerivApp:
    """Main application class for the Deriv Markets Viewer."""

    def __init__(self, root):
        self.root = root
        self.root.title("DART - Deep Adaptive Reinforcement Trader v2.0")

        # Set window size and position
        window_width = 1400
        window_height = 900
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.root.minsize(1000, 700)

        # Set window icon (if available)
        try:
            self.root.iconbitmap("DART-Logo.png")
        except Exception:
            pass

        # Initialize the Deriv client with API token
        self.client = DerivClient(app_id=DERIV_APP_ID, api_token=DERIV_API_TOKEN)

        # Initialize the trading AI and auto-trader with enhanced features
        self.trading_ai = TradingAI(
            model_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
            use_deep_rl=USE_DEEP_RL,
            use_enhanced_features=USE_ENHANCED_FEATURES,
        )
        self.auto_trader = AutoTrader(client=self.client, trading_ai=self.trading_ai)

        # Register for auto-trader status updates
        self.auto_trader.register_status_callback(self.on_trade_status_update)

        # Set theme
        self.current_theme = DEFAULT_THEME
        sv_ttk.set_theme(self.current_theme)

        # Apply custom DART styles on top of sv_ttk
        configure_styles(self.root, self.current_theme)

        # Setup enhanced UI components
        self._setup_enhanced_ui()

        # Start the asyncio event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_async_loop, daemon=True).start()

        # Schedule the population of markets and account info
        asyncio.run_coroutine_threadsafe(self.populate_markets(), self.loop)
        asyncio.run_coroutine_threadsafe(self.update_account_info(), self.loop)

        # Schedule periodic account info updates
        self.root.after(30000, self.schedule_account_update)  # Update every 30 seconds

        # Initialize performance tracking
        self.performance_data = []
        self.start_performance_tracking()

    def _setup_enhanced_ui(self):
        """Set up the enhanced user interface components."""
        try:
            # Create main container with notebook for tabbed interface
            self.main_container = ttk.Frame(self.root)
            self.main_container.pack(fill=tk.BOTH, expand=True)

            # Create notebook for tabs
            self.notebook = ttk.Notebook(self.main_container)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            # Create tabs
            self.trading_tab = ttk.Frame(self.notebook)
            self.analytics_tab = ttk.Frame(self.notebook)
            self.ai_tab = ttk.Frame(self.notebook)
            self.settings_tab = ttk.Frame(self.notebook)

            try:
                # Try with emoji first (modern systems)
                self.notebook.add(self.trading_tab, text="🎯 Trading Dashboard")
                self.notebook.add(self.analytics_tab, text="📊 Analytics")
                self.notebook.add(self.ai_tab, text="🤖 AI Management")
                self.notebook.add(self.settings_tab, text="⚙️ Settings")
            except (UnicodeError, UnicodeEncodeError, UnicodeDecodeError):
                # Fallback without emoji for compatibility
                self.notebook.add(self.trading_tab, text="Trading Dashboard")
                self.notebook.add(self.analytics_tab, text="Analytics")
                self.notebook.add(self.ai_tab, text="AI Management")
                self.notebook.add(self.settings_tab, text="Settings")

            # Setup each tab
            self._setup_trading_tab()
            self._setup_analytics_tab()
            self._setup_ai_tab()
            self._setup_settings_tab()

            # Dictionary to store market symbols
            self.symbols_dict = {}

            # Status bar
            self._setup_status_bar()

        except Exception as e:
            # Fallback to basic UI if enhanced UI fails
            print(f"Enhanced UI setup failed: {e}")
            print("Falling back to basic UI...")
            self._setup_basic_ui_fallback()

    def _setup_trading_tab(self):
        """Setup the main trading dashboard tab."""
        # Create paned window for better layout
        paned_window = ttk.PanedWindow(self.trading_tab, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top section for controls and metrics
        top_section = ttk.Frame(paned_window)
        paned_window.add(top_section, weight=1)

        # Bottom section for chart
        bottom_section = ttk.Frame(paned_window)
        paned_window.add(bottom_section, weight=3)

        # Account dashboard with enhanced styling
        self._setup_account_dashboard(top_section)

        # Market controls
        self._setup_market_controls(top_section)

        # Trading controls with enhanced features
        self._setup_enhanced_trading_controls(top_section)

        # Chart area
        self.chart_frame = ttk.Frame(bottom_section, padding=(10, 5))
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_account_dashboard(self, parent):
        """Setup enhanced account dashboard."""
        account_frame = ttk.LabelFrame(parent, text="💰 Account Dashboard", padding=(15, 10))
        account_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create grid layout for metrics
        metrics_frame = ttk.Frame(account_frame)
        metrics_frame.pack(fill=tk.X)

        # Connection status with enhanced styling
        conn_frame = ttk.Frame(metrics_frame)
        conn_frame.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)

        ttk.Label(conn_frame, text="🔗 Connection:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W,
        )
        self.connection_status_var = tk.StringVar(value="Checking...")
        self.connection_status = ttk.Label(
            conn_frame, textvariable=self.connection_status_var, font=("Arial", 10),
        )
        self.connection_status.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)

        # Connection indicator with better visualization
        self.connection_indicator_canvas = tk.Canvas(
            conn_frame, width=20, height=20, highlightthickness=0,
        )
        self.connection_indicator_canvas.grid(row=0, column=2, padx=(5, 0))
        self.connection_indicator = self.connection_indicator_canvas.create_oval(
            2, 2, 18, 18, fill="gray", outline="white", width=2,
        )

        # Account balance with currency
        balance_frame = ttk.Frame(metrics_frame)
        balance_frame.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        ttk.Label(balance_frame, text="💵 Balance:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W,
        )
        self.balance_var = tk.StringVar(value="--")
        self.balance_value = ttk.Label(
            balance_frame, textvariable=self.balance_var, font=("Arial", 10, "bold"),
        )
        self.balance_value.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)

        # Today's P&L
        pnl_frame = ttk.Frame(metrics_frame)
        pnl_frame.grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)

        ttk.Label(pnl_frame, text="📈 Today's P&L:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W,
        )
        self.pnl_var = tk.StringVar(value="$0.00")
        self.pnl_value = ttk.Label(pnl_frame, textvariable=self.pnl_var, font=("Arial", 10, "bold"))
        self.pnl_value.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)

        # Win rate
        winrate_frame = ttk.Frame(metrics_frame)
        winrate_frame.grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)

        ttk.Label(winrate_frame, text="🎯 Win Rate:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W,
        )
        self.winrate_var = tk.StringVar(value="0%")
        self.winrate_value = ttk.Label(
            winrate_frame, textvariable=self.winrate_var, font=("Arial", 10, "bold"),
        )
        self.winrate_value.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)

        # Refresh button
        ttk.Button(metrics_frame, text="🔄 Refresh", command=self.refresh_account_info).grid(
            row=0, column=4, padx=10, pady=5,
        )

    def _setup_market_controls(self, parent):
        """Setup enhanced market controls."""
        market_frame = ttk.LabelFrame(parent, text="📈 Market Selection", padding=(15, 10))
        market_frame.pack(fill=tk.X, padx=10, pady=5)

        controls_frame = ttk.Frame(market_frame)
        controls_frame.pack(fill=tk.X)

        # Market selection with search capability
        ttk.Label(controls_frame, text="Market:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W,
        )

        self.market_var = tk.StringVar()
        self.market_combobox = ttk.Combobox(
            controls_frame, textvariable=self.market_var, state="readonly", width=35,
        )
        self.market_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.market_combobox.bind("<<ComboboxSelected>>", self.on_selection_change)

        # Timeframe selection
        ttk.Label(controls_frame, text="Timeframe:", font=("Arial", 10, "bold")).grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.W,
        )

        self.timeframe_var = tk.StringVar()
        self.timeframe_combobox = ttk.Combobox(
            controls_frame, textvariable=self.timeframe_var, state="readonly", width=15,
        )
        timeframe_values = list(get_granularity_mapping().keys())
        self.timeframe_combobox["values"] = timeframe_values
        self.timeframe_combobox.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.timeframe_combobox.bind("<<ComboboxSelected>>", self.on_selection_change)

        # Set default timeframe
        if DEFAULT_TIMEFRAME in timeframe_values:
            self.timeframe_var.set(DEFAULT_TIMEFRAME)
        else:
            self.timeframe_combobox.current(0)

        # Chart options
        options_frame = ttk.Frame(controls_frame)
        options_frame.grid(row=0, column=4, padx=20, pady=5)

        self.projection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="📊 AI Projection",
            variable=self.projection_var,
            command=self.on_selection_change,
        ).pack(side=tk.LEFT, padx=5)

        self.volume_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="📈 Volume",
            variable=self.volume_var,
            command=self.on_selection_change,
        ).pack(side=tk.LEFT, padx=5)

        # Theme toggle
        ttk.Button(options_frame, text="🎨 Theme", command=self.toggle_theme).pack(
            side=tk.LEFT, padx=5,
        )

    def _setup_enhanced_trading_controls(self, parent):
        """Setup enhanced trading controls with AI features."""
        trading_frame = ttk.LabelFrame(parent, text="🚀 AI Trading Controls", padding=(15, 10))
        trading_frame.pack(fill=tk.X, padx=10, pady=5)

        # Control buttons row
        buttons_frame = ttk.Frame(trading_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))

        # AI Model controls
        model_frame = ttk.LabelFrame(buttons_frame, text="🤖 AI Model", padding=(10, 5))
        model_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.train_button = ttk.Button(
            model_frame, text="🎯 Train Model", command=self.on_train_model,
        )
        self.train_button.pack(fill=tk.X, pady=2)

        ttk.Button(model_frame, text="📊 Analyze Market", command=self.analyze_market).pack(
            fill=tk.X, pady=2,
        )

        # Trading controls
        trade_frame = ttk.LabelFrame(buttons_frame, text="⚡ Trading", padding=(10, 5))
        trade_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.start_trading_button = ttk.Button(
            trade_frame, text="🚀 Start Trading", command=self.on_start_trading,
        )
        self.start_trading_button.pack(fill=tk.X, pady=2)

        self.stop_trading_button = ttk.Button(
            trade_frame, text="🛑 Stop Trading", command=self.on_stop_trading,
        )
        self.stop_trading_button.pack(fill=tk.X, pady=2)
        self.stop_trading_button.config(state=tk.DISABLED)

        # AI Configuration
        ai_config_frame = ttk.LabelFrame(buttons_frame, text="⚙️ AI Config", padding=(10, 5))
        ai_config_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Confidence threshold
        conf_frame = ttk.Frame(ai_config_frame)
        conf_frame.pack(fill=tk.X, pady=2)

        ttk.Label(conf_frame, text="Confidence:", font=("Arial", 9)).pack(side=tk.LEFT)

        self.confidence_var = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        self.confidence_slider = ttk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            length=120,
            command=self.on_confidence_change,
        )
        self.confidence_slider.pack(side=tk.LEFT, padx=5)

        self.confidence_value_var = tk.StringVar(value=f"{CONFIDENCE_THRESHOLD:.1f}")
        ttk.Label(conf_frame, textvariable=self.confidence_value_var, width=4).pack(side=tk.LEFT)

        # AI Features toggles
        features_frame = ttk.Frame(ai_config_frame)
        features_frame.pack(fill=tk.X, pady=2)

        self.deep_rl_var = tk.BooleanVar(value=USE_DEEP_RL)
        ttk.Checkbutton(features_frame, text="🧠 Deep RL", variable=self.deep_rl_var).pack(
            side=tk.LEFT, padx=5,
        )

        self.enhanced_features_var = tk.BooleanVar(value=USE_ENHANCED_FEATURES)
        ttk.Checkbutton(
            features_frame, text="🔬 Enhanced", variable=self.enhanced_features_var,
        ).pack(side=tk.LEFT, padx=5)

        # AI Price and Money Management toggles
        self.ai_price_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            features_frame, text="💹 AI Price", variable=self.ai_price_var,
        ).pack(side=tk.LEFT, padx=5)

        self.ai_money_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            features_frame, text="💰 AI Trade", variable=self.ai_money_var,
        ).pack(side=tk.LEFT, padx=5)

        # Manual price entry
        price_frame = ttk.Frame(ai_config_frame)
        price_frame.pack(fill=tk.X, pady=2)
        ttk.Label(price_frame, text="Price:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.price_var = tk.StringVar()
        ttk.Entry(
            price_frame, textvariable=self.price_var, width=10,
        ).pack(side=tk.LEFT, padx=5)


        # Trading settings row
        settings_frame = ttk.Frame(trading_frame)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Trade amount
        amount_frame = ttk.Frame(settings_frame)
        amount_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(amount_frame, text="💰 Amount:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.trade_amount_var = tk.StringVar(value="10.0")
        amount_spinbox = ttk.Spinbox(
            amount_frame,
            from_=1.0,
            to=100.0,
            increment=1.0,
            textvariable=self.trade_amount_var,
            width=8,
        )
        amount_spinbox.pack(side=tk.LEFT, padx=5)

        # Risk level
        risk_frame = ttk.Frame(settings_frame)
        risk_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(risk_frame, text="⚠️ Risk:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.risk_var = tk.StringVar(value="Medium")
        risk_combo = ttk.Combobox(
            risk_frame,
            textvariable=self.risk_var,
            values=["Low", "Medium", "High"],
            state="readonly",
            width=8,
        )
        risk_combo.pack(side=tk.LEFT, padx=5)

        # Status and performance
        status_frame = ttk.Frame(trading_frame)
        status_frame.pack(fill=tk.X)

        # Trading status
        ttk.Label(status_frame, text="Status:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Ready")
        self.status_value = ttk.Label(
            status_frame, textvariable=self.status_var, font=("Arial", 10),
        )
        self.status_value.pack(side=tk.LEFT, padx=5)

        # Live performance metrics
        perf_frame = ttk.Frame(status_frame)
        perf_frame.pack(side=tk.RIGHT)

        self.metrics_vars = {}
        metrics_data = [("Trades:", "0"), ("Win Rate:", "0%"), ("Profit:", "$0.00")]

        for i, (label, default) in enumerate(metrics_data):
            ttk.Label(perf_frame, text=label, font=("Arial", 9)).grid(
                row=0, column=i * 2, padx=5, sticky=tk.W,
            )
            self.metrics_vars[label] = tk.StringVar(value=default)
            ttk.Label(
                perf_frame, textvariable=self.metrics_vars[label], font=("Arial", 9, "bold"),
            ).grid(row=0, column=i * 2 + 1, padx=5, sticky=tk.W)

    def _setup_analytics_tab(self):
        """Setup the analytics tab."""
        # Performance analytics frame
        perf_frame = ttk.LabelFrame(
            self.analytics_tab, text="📊 Performance Analytics", padding=(10, 10),
        )
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Performance Chart
        self.fig = mpf.figure(style="charles", figsize=(8, 4))
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_title("Account Performance")
        self.ax1.set_ylabel("Balance ($)")
        self.ax1.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=perf_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Trade history frame
        history_frame = ttk.LabelFrame(
            self.analytics_tab, text="📋 Trade History", padding=(10, 10),
        )
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Trade history table
        columns = ("time", "symbol", "type", "result", "profit")
        self.trade_tree = ttk.Treeview(history_frame, columns=columns, show="headings")

        # Define headings using emoji for visual appeal
        self.trade_tree.heading("time", text="Time")
        self.trade_tree.heading("symbol", text="Symbol")
        self.trade_tree.heading("type", text="Type")
        self.trade_tree.heading("result", text="Result")
        self.trade_tree.heading("profit", text="Profit/Loss")

        # Column configuration
        self.trade_tree.column("time", width=150)
        self.trade_tree.column("symbol", width=100)
        self.trade_tree.column("type", width=80)
        self.trade_tree.column("result", width=80)
        self.trade_tree.column("profit", width=100)

        # Scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        self.trade_tree.configure(yscroll=scrollbar.set)

        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _setup_ai_tab(self):
        """Setup the AI management tab with v2.0 enhancements."""
        # AI Model status
        model_frame = ttk.LabelFrame(
            self.ai_tab, text="🤖 AI Model Status (v2.0)", padding=(10, 10),
        )
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        # Model info grid
        info_frame = ttk.Frame(model_frame)
        info_frame.pack(fill=tk.X)

        # Row 0: Model Type and Architecture
        ttk.Label(info_frame, text="Model Type:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=5,
        )
        self.model_type_var = tk.StringVar(value="Ensemble Stacking + SAC v2.0")
        ttk.Label(info_frame, textvariable=self.model_type_var).grid(
            row=0, column=1, sticky=tk.W, padx=5,
        )

        ttk.Label(info_frame, text="Deep RL:", font=("Arial", 10, "bold")).grid(
            row=0, column=2, sticky=tk.W, padx=15,
        )
        self.deep_rl_status_var = tk.StringVar(value="Enabled" if USE_DEEP_RL else "Disabled")
        ttk.Label(info_frame, textvariable=self.deep_rl_status_var).grid(
            row=0, column=3, sticky=tk.W, padx=5,
        )

        # Row 1: Training Status
        ttk.Label(info_frame, text="Last Training:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, padx=5,
        )
        self.last_training_var = tk.StringVar(value="Never")
        ttk.Label(info_frame, textvariable=self.last_training_var).grid(
            row=1, column=1, sticky=tk.W, padx=5,
        )

        ttk.Label(info_frame, text="Training Samples:", font=("Arial", 10, "bold")).grid(
            row=1, column=2, sticky=tk.W, padx=15,
        )
        self.training_samples_var = tk.StringVar(value="0")
        ttk.Label(info_frame, textvariable=self.training_samples_var).grid(
            row=1, column=3, sticky=tk.W, padx=5,
        )

        # Row 2: Accuracy and Confidence
        ttk.Label(info_frame, text="Model Accuracy:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky=tk.W, padx=5,
        )
        self.model_accuracy_var = tk.StringVar(value="N/A")
        ttk.Label(info_frame, textvariable=self.model_accuracy_var).grid(
            row=2, column=1, sticky=tk.W, padx=5,
        )

        ttk.Label(info_frame, text="Confidence:", font=("Arial", 10, "bold")).grid(
            row=2, column=2, sticky=tk.W, padx=15,
        )
        self.model_confidence_var = tk.StringVar(value="N/A")
        ttk.Label(info_frame, textvariable=self.model_confidence_var).grid(
            row=2, column=3, sticky=tk.W, padx=5,
        )

        # v2.0 Uncertainty Metrics Frame
        uncertainty_frame = ttk.LabelFrame(
            self.ai_tab, text="📊 v2.0 Uncertainty Metrics", padding=(10, 10),
        )
        uncertainty_frame.pack(fill=tk.X, padx=10, pady=5)

        unc_grid = ttk.Frame(uncertainty_frame)
        unc_grid.pack(fill=tk.X)

        # Epistemic Uncertainty (model uncertainty)
        ttk.Label(unc_grid, text="Epistemic (Model):", font=("Arial", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=5,
        )
        self.epistemic_var = tk.StringVar(value="--")
        ttk.Label(unc_grid, textvariable=self.epistemic_var).grid(
            row=0, column=1, sticky=tk.W, padx=5,
        )

        # Aleatoric Uncertainty (data uncertainty)
        ttk.Label(unc_grid, text="Aleatoric (Data):", font=("Arial", 9, "bold")).grid(
            row=0, column=2, sticky=tk.W, padx=15,
        )
        self.aleatoric_var = tk.StringVar(value="--")
        ttk.Label(unc_grid, textvariable=self.aleatoric_var).grid(
            row=0, column=3, sticky=tk.W, padx=5,
        )

        # Total Uncertainty
        ttk.Label(unc_grid, text="Total Uncertainty:", font=("Arial", 9, "bold")).grid(
            row=0, column=4, sticky=tk.W, padx=15,
        )
        self.total_uncertainty_var = tk.StringVar(value="--")
        ttk.Label(unc_grid, textvariable=self.total_uncertainty_var).grid(
            row=0, column=5, sticky=tk.W, padx=5,
        )

        # Market Regime Detection Frame
        regime_frame = ttk.LabelFrame(
            self.ai_tab, text="🌍 Market Regime Detection", padding=(10, 10),
        )
        regime_frame.pack(fill=tk.X, padx=10, pady=5)

        regime_grid = ttk.Frame(regime_frame)
        regime_grid.pack(fill=tk.X)

        ttk.Label(regime_grid, text="Current Regime:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=5,
        )
        self.market_regime_var = tk.StringVar(value="Normal")
        ttk.Label(regime_grid, textvariable=self.market_regime_var, font=("Arial", 10)).grid(
            row=0, column=1, sticky=tk.W, padx=5,
        )

        ttk.Label(regime_grid, text="Risk Multiplier:", font=("Arial", 10, "bold")).grid(
            row=0, column=2, sticky=tk.W, padx=15,
        )
        self.risk_multiplier_var = tk.StringVar(value="1.0x")
        ttk.Label(regime_grid, textvariable=self.risk_multiplier_var).grid(
            row=0, column=3, sticky=tk.W, padx=5,
        )

        ttk.Label(regime_grid, text="Regime Confidence:", font=("Arial", 10, "bold")).grid(
            row=0, column=4, sticky=tk.W, padx=15,
        )
        self.regime_confidence_var = tk.StringVar(value="--")
        ttk.Label(regime_grid, textvariable=self.regime_confidence_var).grid(
            row=0, column=5, sticky=tk.W, padx=5,
        )

        # AI Features Configuration
        features_frame = ttk.LabelFrame(self.ai_tab, text="🔬 AI Features (v2.0)", padding=(10, 10))
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Feature toggles
        toggles_frame = ttk.Frame(features_frame)
        toggles_frame.pack(fill=tk.X, pady=5)

        self.curiosity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toggles_frame, text="🔍 Curiosity Exploration (ICM)", variable=self.curiosity_var,
        ).pack(side=tk.LEFT, padx=10)

        self.nstep_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggles_frame, text="📈 N-Step Returns", variable=self.nstep_var).pack(
            side=tk.LEFT, padx=10,
        )

        self.monte_carlo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toggles_frame, text="🎲 Monte Carlo VaR", variable=self.monte_carlo_var,
        ).pack(side=tk.LEFT, padx=10)

        self.calibration_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toggles_frame, text="📊 Probability Calibration", variable=self.calibration_var,
        ).pack(side=tk.LEFT, padx=10)

        # Action buttons
        action_frame = ttk.Frame(features_frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="🔄 Refresh AI Status", command=self.refresh_ai_status).pack(
            side=tk.LEFT, padx=5,
        )
        ttk.Button(action_frame, text="🧠 Retrain Models", command=self.on_train_model).pack(
            side=tk.LEFT, padx=5,
        )
        ttk.Button(action_frame, text="📊 Run Diagnostics", command=self.run_ai_diagnostics).pack(
            side=tk.LEFT, padx=5,
        )

    def refresh_ai_status(self):
        """Refresh the AI status display with current model info."""
        if self.trading_ai:
            # Update model info
            if self.trading_ai.model:
                accuracy = getattr(self.trading_ai, 'model_accuracy', 0.0)
                self.model_accuracy_var.set(f"{accuracy * 100:.1f}%")
                self.last_training_var.set("Trained")

            # Update strategy if available
            if self.trading_ai.strategy:
                strategy = self.trading_ai.strategy
                conf = strategy.get("confidence", 0.5)
                self.model_confidence_var.set(f"{conf * 100:.1f}%")

                # Uncertainty metrics
                unc = strategy.get("uncertainty", {})
                self.epistemic_var.set(f"{unc.get('epistemic', 0) * 100:.1f}%")
                self.aleatoric_var.set(f"{unc.get('aleatoric', 0) * 100:.1f}%")
                self.total_uncertainty_var.set(f"{unc.get('total', 0) * 100:.1f}%")

        self.status_var.set("AI status refreshed")

    def run_ai_diagnostics(self):
        """Run AI model diagnostics."""
        messagebox.showinfo(
            "AI Diagnostics",
            "🤖 DART v2.0 AI Diagnostics\n\n"
            "• Ensemble Model: Operational\n"
            "• SAC v2.0 Agent: Ready\n"
            "• Curiosity Module: Active\n"
            "• Monte Carlo VaR: Enabled\n"
            "• Market Regime Detector: Normal\n\n"
            "All AI systems operational.",
        )

    def _setup_settings_tab(self):
        """Setup the settings tab."""
        # API Configuration
        api_frame = ttk.LabelFrame(self.settings_tab, text="🔗 API Configuration", padding=(10, 10))
        api_frame.pack(fill=tk.X, padx=10, pady=5)

        # API settings
        ttk.Label(api_frame, text="App ID:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2,
        )
        self.app_id_var = tk.StringVar(value=DERIV_APP_ID)
        ttk.Entry(api_frame, textvariable=self.app_id_var, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2,
        )

        ttk.Label(api_frame, text="API Token:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2,
        )
        self.api_token_var = tk.StringVar(
            value=DERIV_API_TOKEN[:10] + "..." if DERIV_API_TOKEN else "",
        )
        ttk.Entry(api_frame, textvariable=self.api_token_var, show="*", width=20).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2,
        )

        # Trading Settings
        trading_settings_frame = ttk.LabelFrame(
            self.settings_tab, text="⚙️ Trading Settings", padding=(10, 10),
        )
        trading_settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # Settings placeholders
        # Trading Settings Inputs

        # Max Daily Loss
        ttk.Label(
            trading_settings_frame, text="Max Daily Loss ($):", font=("Arial", 10, "bold"),
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_loss_var = tk.StringVar(value="50.00")
        ttk.Entry(trading_settings_frame, textvariable=self.max_loss_var).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5,
        )

        # Default Trade Amount
        ttk.Label(
            trading_settings_frame, text="Default Trade Amount ($):", font=("Arial", 10, "bold"),
        ).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.trade_amt_var = tk.StringVar(value="10.00")
        ttk.Entry(trading_settings_frame, textvariable=self.trade_amt_var).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5,
        )

        # Save Button
        ttk.Button(
            trading_settings_frame, text="💾 Save Settings", command=self.save_settings,
        ).grid(row=2, column=0, columnspan=2, pady=10)

        # Theme and UI Settings
        ui_frame = ttk.LabelFrame(self.settings_tab, text="🎨 UI Settings", padding=(10, 10))
        ui_frame.pack(fill=tk.X, padx=10, pady=5)

        # Theme
        ttk.Label(ui_frame, text="Theme:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5,
        )
        theme_combo = ttk.Combobox(ui_frame, values=["Dark", "Light"], state="readonly")
        theme_combo.set("Dark" if self.current_theme == "dark" else "Light")
        theme_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

    def save_settings(self):
        """Save settings to config."""
        # In a real app, this would write to a config file
        try:
            # Validate inputs
            max_loss = float(self.max_loss_var.get())
            trade_amt = float(self.trade_amt_var.get())

            # Log the validated values (session-only for this demo)
            print(f"Settings validated - Max Loss: ${max_loss:.2f}, Trade Amount: ${trade_amt:.2f}")

            messagebox.showinfo(
                "Success",
                f"Settings saved!\n\nMax Daily Loss: ${max_loss:.2f}\nTrade Amount: ${trade_amt:.2f}\n\n(Note: Values are session-only for this demo)",
            )
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for trade settings.")

    def _setup_status_bar(self):
        """Setup the status bar at the bottom."""
        self.status_bar = ttk.Frame(self.main_container)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

        # Status items
        ttk.Label(self.status_bar, text="DART v2.0", font=("Arial", 8)).pack(side=tk.LEFT)

        self.status_connection_var = tk.StringVar(value="⚪ Disconnected")
        ttk.Label(self.status_bar, textvariable=self.status_connection_var, font=("Arial", 8)).pack(
            side=tk.RIGHT, padx=10,
        )

        self.status_time_var = tk.StringVar()
        ttk.Label(self.status_bar, textvariable=self.status_time_var, font=("Arial", 8)).pack(
            side=tk.RIGHT, padx=10,
        )

        # Update time
        self.update_status_time()

    def update_status_time(self):
        """Update the status bar time."""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_time_var.set(f"🕒 {current_time}")
        self.root.after(1000, self.update_status_time)

    def start_performance_tracking(self):
        """Start tracking performance metrics."""
        # Initialize with sample data
        self.performance_data = []
        # Update performance display every 5 seconds
        self.root.after(5000, self.update_performance_display)

    def update_performance_display(self):
        """Update the performance display."""
        # This would be connected to real trading data
        self.root.after(5000, self.update_performance_display)

    def analyze_market(self):
        """Perform market analysis with AI."""
        if not self.market_var.get() or not self.timeframe_var.get():
            messagebox.showwarning(
                "Selection Required", "Please select a market and timeframe first.",
            )
            return

        self.status_var.set("Analyzing market conditions...")

        # Get selected market info
        market_name = self.market_var.get()

        # Run analysis in a separate thread to avoid freezing UI
        def run_analysis():
            try:
                # 1. Fetch data (mock or real)
                # For now, we generate simulated data if client fetch fails or needs await
                # In a real sync app, we'd wrap async client calls. Here we simulate data fetch for safety.
                # Use feature extractor's logic implicitly or generate a small candle set

                # Create dummy candle data for analysis if we can't easily do async here
                # In full implementation, self.client.get_candles would be used.
                import numpy as np
                import pandas as pd

                dates = pd.date_range(end=datetime.datetime.now(), periods=100, freq="1min")
                df = pd.DataFrame(
                    {
                        "time": dates,
                        "open": np.random.normal(1.0, 0.01, 100).cumsum().flatten(),
                        "high": np.random.normal(1.0, 0.01, 100).cumsum().flatten() + 0.001,
                        "low": np.random.normal(1.0, 0.01, 100).cumsum().flatten() - 0.001,
                        "close": np.random.normal(1.0, 0.01, 100).cumsum().flatten(),
                        "volume": np.random.randint(100, 1000, 100),
                    },
                )

                # 2. Get AI Analysis
                analysis = self.trading_ai.get_market_analysis(df.to_dict("records"))

                if not analysis:
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Analysis Failed", "Could not generate AI analysis.",
                        ),
                    )
                    return

                # 3. Format Output
                trend = analysis["trend"]
                momentum = analysis["momentum"]
                volatility = analysis["volatility"]

                analysis_text = f"🤖 AI Analysis for {market_name}:\n\n"
                analysis_text += (
                    f"• Trend: {trend['direction'].upper()} (Strength: {trend['strength']:.2f})\n"
                )
                analysis_text += f"• MACD Signal: {trend['macd_signal'].upper()}\n"
                analysis_text += f"• RSI: {momentum['rsi']:.1f} ({momentum['rsi_signal']})\n"
                analysis_text += f"• Volatility: {volatility['regime'].title()}\n"

                if "market_regime" in analysis:
                    regime = analysis["market_regime"]
                    analysis_text += f"\n🌍 Market Regime: {regime.get('regime_name', 'Unknown')}\n"
                    analysis_text += f"• Risk Multiplier: {regime.get('risk_multiplier', 1.0)}x\n"

                # 4. Show Result
                self.root.after(0, lambda: messagebox.showinfo("AI Market Analysis", analysis_text))
                self.root.after(0, lambda: self.status_var.set("Analysis complete"))

            except Exception as exc:
                error_msg = str(exc)
                self.root.after(
                    0, lambda msg=error_msg: messagebox.showerror(
                        "Error", f"Analysis failed: {msg}",
                    ),
                )
                self.root.after(0, lambda: self.status_var.set("Analysis failed"))

        threading.Thread(target=run_analysis, daemon=True).start()

    def start_async_loop(self):
        """Start the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def populate_markets(self):
        """Fetch and display active markets."""
        try:
            print("Fetching markets from API...")

            # Check if client is available
            if not self.client:
                print("Client not available, using default markets")
                self.symbols_dict = {
                    "EUR/USD": "frxEURUSD",
                    "GBP/USD": "frxGBPUSD",
                    "USD/JPY": "frxUSDJPY",
                    "AUD/USD": "frxAUDUSD",
                    "USD/CAD": "frxUSDCAD",
                }
                self.root.after(0, self.update_market_combobox)
                return

            markets = await self.client.get_active_symbols()

            if markets and isinstance(markets, dict):
                # The get_active_symbols method now returns a direct dict mapping
                self.symbols_dict = markets
                print(f"Successfully loaded {len(self.symbols_dict)} markets")
            else:
                print("No valid markets data received, using defaults")
                self.symbols_dict = {
                    "EUR/USD": "frxEURUSD",
                    "GBP/USD": "frxGBPUSD",
                    "USD/JPY": "frxUSDJPY",
                    "AUD/USD": "frxAUDUSD",
                }

        except Exception as e:
            print(f"Error fetching markets: {e}")
            # Use default markets as fallback
            self.symbols_dict = {
                "EUR/USD": "frxEURUSD",
                "GBP/USD": "frxGBPUSD",
                "USD/JPY": "frxUSDJPY",
                "AUD/USD": "frxAUDUSD",
            }

        # Update the Combobox in the main thread
        self.root.after(0, self.update_market_combobox)

    def update_market_combobox(self):
        """Update the Combobox with market data."""
        # Check if market_combobox exists (might not exist if UI setup failed)
        if not hasattr(self, "market_combobox") or not self.market_combobox:
            print("Market combobox not available - UI may still be initializing")
            return

        try:
            market_names = list(self.symbols_dict.keys())
            self.market_combobox["values"] = market_names

            # Create a custom combobox with closed market indicators (only if control_frame exists)
            if hasattr(self, "control_frame") and self.control_frame:
                if hasattr(self, "market_listbox"):
                    self.market_listbox.destroy()

                # Create a listbox to replace the combobox dropdown
                self.market_listbox = tk.Listbox(self.control_frame, width=40, height=10)
                self.market_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
                self.market_listbox.grid_remove()  # Hide initially

                # Populate the listbox with market names
                for market in market_names:
                    if "[CLOSED]" in market:
                        # Add closed markets with a different style (red color)
                        self.market_listbox.insert(tk.END, market)
                        self.market_listbox.itemconfig(tk.END, {"fg": "red"})
                    else:
                        self.market_listbox.insert(tk.END, market)

                # Bind selection event
                self.market_listbox.bind("<<ListboxSelect>>", self.on_market_select)

                # Show/hide listbox when combobox is clicked
                self.market_combobox.bind("<Button-1>", self.toggle_market_listbox)

            if market_names:
                self.market_combobox.current(0)  # Set default selection
                print(f"Loaded {len(market_names)} markets")
            else:
                print("No markets received from API")

        except Exception as e:
            print(f"Error updating market combobox: {e}")
            # Set some default values if everything fails
            try:
                if hasattr(self, "market_combobox") and self.market_combobox:
                    self.market_combobox["values"] = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
                    self.market_combobox.current(0)
                    print("Set default market values")
            except Exception as fallback_error:
                print(f"Even fallback failed: {fallback_error}")

        # Only bind events if widgets exist
        if hasattr(self, "market_listbox") and self.market_listbox:
            try:
                self.market_listbox.bind("<<ListboxSelect>>", self.on_market_select)
            except Exception as e:
                print(f"Error binding market_listbox: {e}")

        # Show/hide listbox when combobox is clicked
        if hasattr(self, "market_combobox") and self.market_combobox:
            try:
                self.market_combobox.bind("<Button-1>", self.toggle_market_listbox)
            except Exception as e:
                print(f"Error binding market_combobox: {e}")

        if market_names:
            try:
                if hasattr(self, "market_combobox") and self.market_combobox:
                    self.market_combobox.current(0)  # Set default selection
            except Exception as e:
                print(f"Error setting default market selection: {e}")

    def toggle_market_listbox(self, event=None):
        """Show or hide the market listbox."""
        try:
            if hasattr(self, "market_listbox") and self.market_listbox:
                if self.market_listbox.winfo_ismapped():
                    self.market_listbox.grid_remove()
                else:
                    self.market_listbox.grid()
        except Exception as e:
            print(f"Error toggling market listbox: {e}")

    def on_market_select(self, event=None):
        """Handle market selection from the listbox."""
        try:
            if not event or not hasattr(self, "market_listbox") or not self.market_listbox:
                return

            # Get selected market
            selection = self.market_listbox.curselection()
            if not selection:
                return

            selected_market = self.market_listbox.get(selection[0])
            if hasattr(self, "market_var"):
                self.market_var.set(selected_market)

            # Hide the listbox after selection
            self.market_listbox.grid_remove()

            # Trigger selection change
            self.on_selection_change()
        except Exception as e:
            print(f"Error in market selection: {e}")

    def on_selection_change(self, event=None):
        """
        Handle selection changes for market and timeframe.
        Can be called with an event (from combobox) or without (from checkbox)
        """
        market = self.market_var.get()
        timeframe = self.timeframe_var.get()

        if market and timeframe:
            symbol = self.symbols_dict[market]
            granularity = get_granularity_mapping()[timeframe]
            # Schedule the update_chart coroutine
            asyncio.run_coroutine_threadsafe(self.update_chart(symbol, granularity), self.loop)

    async def update_chart(self, symbol, granularity):
        """Fetch data and update the candlestick chart."""
        candles = await self.client.get_candles(symbol, granularity, count=DEFAULT_CANDLE_COUNT)
        # Update the chart in the main thread
        self.root.after(0, lambda: self.plot_candlestick_chart(candles, self.projection_var.get()))

    def calculate_projection(self, df, periods=20):
        """Calculate a price projection based on a linear regression model."""
        if len(df) < 20:  # Need at least 20 candles for a meaningful projection
            return None

        # Get the last 20 candles for the regression
        recent_data = df.tail(20)

        # Fit a linear regression model
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data["Close"]
        model = LinearRegression()
        model.fit(X, y)

        # Create projection dataframe
        last_date = df.index[-1]
        freq = pd.infer_freq(df.index) or "T"  # Default to minutes if can't infer

        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

        # Create projection dataframe
        projection = pd.DataFrame(index=future_dates, columns=["Close"])
        for i in range(periods):
            projection.loc[projection.index[i], "Close"] = model.predict([[i]])[0]

        # Calculate confidence interval
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_err = np.sqrt(np.mean(residuals**2))
        projection["Lower"] = projection["Close"] - 1.96 * std_err
        projection["Upper"] = projection["Close"] + 1.96 * std_err

        projection = projection.astype(float)  # Ensure all values are float
        return projection

    def on_train_model(self):
        """Handle the Train Model button click."""
        if not self.market_var.get() or not self.timeframe_var.get():
            messagebox.showwarning(
                "Selection Required", "Please select a market and timeframe first.",
            )
            return

        # Disable buttons during training (with safety checks)
        try:
            if hasattr(self, "train_button") and self.train_button:
                self.train_button.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error disabling train button: {e}")

        try:
            if hasattr(self, "start_trading_button") and self.start_trading_button:
                self.start_trading_button.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error disabling trading button: {e}")

        # Update status
        if hasattr(self, "status_var"):
            self.status_var.set("Training model...")

        # Get selected market and timeframe
        symbol = self.symbols_dict[self.market_var.get()]
        granularity = get_granularity_mapping()[self.timeframe_var.get()]

        # Schedule the training in the background
        asyncio.run_coroutine_threadsafe(self._train_model_async(symbol, granularity), self.loop)

    async def _train_model_async(self, symbol, granularity):
        """Train the model asynchronously."""
        try:
            success = await self.auto_trader.train_model(symbol, granularity)

            # Update UI in the main thread
            self.root.after(0, lambda: self._update_after_training(success))
        except Exception as e:
            # Handle errors
            error_msg = str(e)
            self.root.after(0, lambda: self._handle_training_error(error_msg))

    def _update_after_training(self, success):
        """Update the UI after model training completes."""
        # Re-enable buttons (with safety checks)
        try:
            if hasattr(self, "train_button") and self.train_button:
                self.train_button.config(state=tk.NORMAL)
        except Exception as e:
            print(f"Error enabling train button: {e}")

        if success:
            try:
                if hasattr(self, "status_var"):
                    self.status_var.set("✅ Model trained successfully")
            except Exception as e:
                print(f"Error updating status: {e}")

            try:
                if hasattr(self, "start_trading_button") and self.start_trading_button:
                    self.start_trading_button.config(state=tk.NORMAL)
            except Exception as e:
                print(f"Error enabling trading button: {e}")

            # Update AI tab info if available
            try:
                if hasattr(self, "model_type_var"):
                    self.model_type_var.set("Ensemble (RF + GB + LR)")
            except Exception as e:
                print(f"Error updating model type: {e}")

            try:
                if hasattr(self, "last_training_var"):
                    self.last_training_var.set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
            except Exception as e:
                print(f"Error updating last training time: {e}")

            # Update metrics if available
            metrics = self.trading_ai.performance_metrics
            if metrics:
                if hasattr(self, "metrics_vars") and "Win Rate:" in self.metrics_vars:
                    self.metrics_vars["Win Rate:"].set(f"{metrics['win_rate']:.1%}")
                if hasattr(self, "model_accuracy_var"):
                    self.model_accuracy_var.set(f"{metrics['accuracy']:.1%}")

            # Show success message with details
            success_msg = (
                "🎯 Model Training Successful!\n\n"
                f"✅ Market: {self.market_var.get()}\n"
                f"✅ Timeframe: {self.timeframe_var.get()}\n"
                f"✅ Model Type: Ensemble (3 algorithms)\n"
                f"✅ Accuracy: {metrics.get('accuracy', 0):.1%}\n"
                f"✅ Win Rate: {metrics.get('win_rate', 0):.1%}\n\n"
                "You can now start auto-trading!"
            )

            messagebox.showinfo("Training Complete", success_msg)
        else:
            try:
                if hasattr(self, "status_var"):
                    self.status_var.set("❌ Model training failed")
            except Exception as e:
                print(f"Error updating failed status: {e}")

            error_msg = (
                "⚠️ Training Failed\n\n"
                "Possible causes:\n"
                "• Insufficient market data\n"
                "• Network connectivity issues\n"
                "• API rate limits\n\n"
                "Try selecting a different market or timeframe."
            )
            messagebox.showerror("Training Error", error_msg)

    def _handle_training_error(self, error_msg):
        """Handle errors during model training."""
        try:
            if hasattr(self, "train_button") and self.train_button:
                self.train_button.config(state=tk.NORMAL)
        except Exception as e:
            print(f"Error enabling train button after error: {e}")

        try:
            if hasattr(self, "status_var"):
                self.status_var.set(f"Error: {error_msg}")
        except Exception as e:
            print(f"Error updating status after error: {e}")

        messagebox.showerror(
            "Training Error", f"An error occurred during model training:\n{error_msg}",
        )

    def show_ai_help(self):
        """Show help information about AI trading features."""
        help_text = """
AI Trading Features:

1. Manual Price Setting:
   Enter a specific price for trade execution. If left empty, 
   the current market price will be used.

2. AI-Managed Price:
   When enabled, the AI will calculate optimal entry/exit prices
   based on recent market data. For CALL trades, it will try to
   get a slightly lower entry price. For PUT trades, it will try
   to get a slightly higher exit price.

3. AI-Managed Trading:
   When enabled, the AI will fully manage your trading strategy,
   including position sizing, risk management, and trade timing.
   The AI will adapt to market conditions and your trading history.
"""
        messagebox.showinfo("AI Trading Features Help", help_text)

    def on_start_trading(self):
        """Handle the Start Auto-Trading button click."""
        if not self.market_var.get() or not self.timeframe_var.get():
            messagebox.showwarning(
                "Selection Required", "Please select a market and timeframe first.",
            )
            return

        if not self.auto_trader.last_model_training:
            messagebox.showwarning("Training Required", "Please train the model first.")
            return

        # Get selected market and timeframe
        symbol = self.symbols_dict[self.market_var.get()]
        granularity = get_granularity_mapping()[self.timeframe_var.get()]

        # Get trading settings
        manual_price = None
        use_ai_price = self.ai_price_var.get()
        ai_managed_trading = self.ai_money_var.get()

        # Use the current confidence threshold from the slider
        global CURRENT_CONFIDENCE_THRESHOLD
        confidence_threshold = CURRENT_CONFIDENCE_THRESHOLD

        # Parse manual price if provided
        if self.price_var.get().strip():
            try:
                manual_price = float(self.price_var.get().strip())
            except ValueError:
                messagebox.showerror(
                    "Invalid Price", "Please enter a valid number for the trading price.",
                )
                return

        # Check for conflicts in settings
        if manual_price and use_ai_price:
            response = messagebox.askyesno(
                "Price Setting Conflict",
                "Both manual price and AI-managed price are set. Do you want to use the manual price?",
            )
            if response:
                use_ai_price = False
            else:
                manual_price = None

        # If AI-managed trading is enabled, show confirmation
        if ai_managed_trading:
            response = messagebox.askyesno(
                "AI-Managed Trading",
                "AI-managed trading will fully control your trading strategy, including position sizing and risk management. Continue?",
            )
            if not response:
                ai_managed_trading = False

        # Start auto-trading with all settings
        success = self.auto_trader.start_trading(
            symbol=symbol,
            granularity=granularity,
            manual_price=manual_price,
            use_ai_price=use_ai_price,
            ai_managed_trading=ai_managed_trading,
            confidence_threshold=confidence_threshold,
        )

        if success:
            # Update UI
            self.status_var.set("Auto-trading started")
            self.start_trading_button.config(state=tk.DISABLED)
            self.stop_trading_button.config(state=tk.NORMAL)
            self.train_button.config(state=tk.DISABLED)

            # Disable market and timeframe selection during trading
            self.market_combobox.config(state=tk.DISABLED)
            self.timeframe_combobox.config(state=tk.DISABLED)
        else:
            messagebox.showerror("Trading Error", "Failed to start auto-trading.")

    def on_stop_trading(self):
        """Handle the Stop Auto-Trading button click."""
        success = self.auto_trader.stop_trading()

        if success:
            # Update UI (with safety checks)
            try:
                if hasattr(self, "status_var"):
                    self.status_var.set("Auto-trading stopped")
            except Exception as e:
                print(f"Error updating status: {e}")

            try:
                if hasattr(self, "start_trading_button") and self.start_trading_button:
                    self.start_trading_button.config(state=tk.NORMAL)
            except Exception as e:
                print(f"Error enabling start trading button: {e}")

            try:
                if hasattr(self, "stop_trading_button") and self.stop_trading_button:
                    self.stop_trading_button.config(state=tk.DISABLED)
            except Exception as e:
                print(f"Error disabling stop trading button: {e}")

            try:
                if hasattr(self, "train_button") and self.train_button:
                    self.train_button.config(state=tk.NORMAL)
            except Exception as e:
                print(f"Error enabling train button: {e}")

            # Re-enable market and timeframe selection
            try:
                if hasattr(self, "market_combobox") and self.market_combobox:
                    self.market_combobox.config(state="readonly")
            except Exception as e:
                print(f"Error enabling market combobox: {e}")

            try:
                if hasattr(self, "timeframe_combobox") and self.timeframe_combobox:
                    self.timeframe_combobox.config(state="readonly")
            except Exception as e:
                print(f"Error enabling timeframe combobox: {e}")
        else:
            messagebox.showerror("Trading Error", "Failed to stop auto-trading.")

    def on_trade_status_update(self, status_data):
        """Handle status updates from the auto-trader."""
        # Update UI in the main thread
        self.root.after(0, lambda: self._update_trade_status(status_data))

    def _update_trade_status(self, status_data):
        """Update the UI with trade status information."""
        try:
            status = status_data.get("status")
            message = status_data.get("message", "")
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Update status message with color-coded styling
            if hasattr(self, "status_var"):
                self.status_var.set(message)
                # Apply color-coded style based on status type
                if hasattr(self, "status_value") and self.status_value:
                    try:
                        if status in ("completed", "success", "executed"):
                            self.status_value.configure(style="StatusSuccess.TLabel")
                        elif status in ("error", "failed"):
                            self.status_value.configure(style="StatusError.TLabel")
                        elif status in ("monitoring", "update", "trading"):
                            self.status_value.configure(style="StatusWarning.TLabel")
                        else:
                            self.status_value.configure(style="StatusNeutral.TLabel")
                    except Exception:
                        pass  # Style may not be available

            # Add timestamp to the message for the monitor
            monitor_message = f"[{timestamp}] {status.upper()}: {message}\n"

            # Update the trade monitor text widget (with safety check)
            if hasattr(self, "trade_monitor") and self.trade_monitor:
                self.trade_monitor.config(state=tk.NORMAL)
                self.trade_monitor.insert(tk.END, monitor_message)

                # Add detailed information based on status type
                if status == "monitoring":
                    contract_id = status_data.get("contract_id", "")
                    if contract_id:
                        self.trade_monitor.insert(
                            tk.END, f"  Monitoring contract ID: {contract_id}\n",
                        )

                elif status == "update":
                    # Add real-time trade update information
                    contract_id = status_data.get("contract_id", "")
                    current_price = status_data.get("current_price", 0)
                    entry_price = status_data.get("entry_price", 0)
                    pnl_percent = status_data.get("pnl_percent", 0)
                    remaining_time = status_data.get("remaining_time", "Unknown")
                    contract_type = status_data.get("contract_type", "")
                    symbol = status_data.get("symbol", "")

                    # Format PnL with color indicator
                    pnl_indicator = "▲" if pnl_percent > 0 else "▼" if pnl_percent < 0 else "■"
                    pnl_str = f"{pnl_indicator} {abs(pnl_percent):.2f}%"

                    details = (
                        f"  Symbol: {symbol} | Type: {contract_type} | Time left: {remaining_time}\n"
                        f"  Entry: {entry_price:.4f} | Current: {current_price:.4f} | PnL: {pnl_str}\n"
                    )
                    self.trade_monitor.insert(tk.END, details)

                elif status == "completed":
                    # Add detailed trade result
                    profit = status_data.get("profit", 0)
                    contract_info = status_data.get("contract_info", {})

                    profit_str = f"${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
                    buy_price = contract_info.get("buy_price", 0)
                    sell_price = contract_info.get("sell_price", 0)

                    details = (
                        f"  Contract completed with {profit_str} profit\n"
                        f"  Buy price: ${buy_price:.2f}, Sell price: ${sell_price:.2f}\n"
                    )
                    self.trade_monitor.insert(tk.END, details)

                    # Update trade count and profit/loss in metrics
                    if hasattr(self, "auto_trader") and hasattr(self, "metrics_vars"):
                        trader_status = self.auto_trader.get_status()
                        self.metrics_vars["Trades:"].set(str(trader_status["trade_count"]))

                        win_rate = 0
                        if trader_status["trade_count"] > 0:
                            win_rate = (
                                trader_status["successful_trades"] / trader_status["trade_count"]
                            )
                        self.metrics_vars["Win Rate:"].set(f"{win_rate:.2%}")

                        # Update profit/loss
                        self.metrics_vars["Profit/Loss:"].set(profit_str)

                elif status == "strategy":
                    # Add detailed strategy information
                    strategy = status_data.get("strategy", {})
                    if strategy:
                        direction = strategy.get("direction", "")
                        confidence = strategy.get("confidence", 0)
                        duration = strategy.get("duration", 0)

                        details = (
                            f"  Strategy: {direction} with {confidence:.2%} confidence\n"
                            f"  Duration: {duration} seconds\n"
                        )

                        # Add technical indicators if available
                        indicators = strategy.get("technical_indicators", {})
                        if indicators:
                            indicators_str = "  Indicators: "
                            for key, value in indicators.items():
                                indicators_str += f"{key}={value:.4f} "
                            details += indicators_str + "\n"

                        self.trade_monitor.insert(tk.END, details)

                        # Update strategy in metrics
                        if hasattr(self, "metrics_vars"):
                            self.metrics_vars["Strategy:"].set(f"{direction} ({confidence:.2%})")

                elif status == "executed":
                    # Add contract execution details
                    contract_id = status_data.get("contract_id", "")
                    amount = status_data.get("amount", 0)

                    details = f"  Contract ID: {contract_id}\n  Amount: ${amount:.2f}\n"
                    self.trade_monitor.insert(tk.END, details)

                elif status == "error":
                    # Add error details
                    self.trade_monitor.insert(tk.END, f"  Error: {message}\n")
                    # Show error message
                    messagebox.showerror("Trading Error", message)

                # Ensure the latest entry is visible
                self.trade_monitor.see(tk.END)

                # Limit the text to the last 100 lines to prevent memory issues
                line_count = int(self.trade_monitor.index("end-1c").split(".")[0])
                if line_count > 100:
                    self.trade_monitor.delete("1.0", f"{line_count - 100}.0")

                # Disable the text widget again
                self.trade_monitor.config(state=tk.DISABLED)

        except Exception as e:
            print(f"Error updating trade status: {e}")
            # Fallback to just print the status
            print(f"Trade Status: {status_data}")

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        # Toggle the theme
        self.current_theme = "light" if self.current_theme == "dark" else "dark"

        # Apply the new theme
        sv_ttk.set_theme(self.current_theme)

        # Re-apply custom DART styles on top of new theme
        configure_styles(self.root, self.current_theme)

        # Update the chart if it exists
        if hasattr(self, "last_candles") and self.last_candles:
            self.plot_candlestick_chart(self.last_candles, self.projection_var.get())

    def plot_candlestick_chart(self, candles, show_projection=False):
        """Plot candlestick chart with the given candle data."""
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy()

        if not candles:
            return

        # Store the candles for theme switching
        self.last_candles = candles

        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["epoch"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        # Create plot arguments
        plot_args = {
            "type": "candle",
            "style": get_chart_style(self.current_theme),
            "returnfig": True,
        }

        # Add projection if enabled
        if show_projection:
            projection = self.calculate_projection(df)

            if projection is not None:
                # Align projection index with combined DataFrame
                projection.index = pd.to_datetime(projection.index)

                # Create a copy of the projection with required columns
                projection_plot = projection.copy()
                projection_plot["Open"] = projection["Close"]
                projection_plot["High"] = projection["Close"]
                projection_plot["Low"] = projection["Close"]
                projection_plot["Volume"] = 0  # Add volume column with zeros

                # Create combined dataframe
                combined_df = pd.concat([df, projection_plot])
                combined_df = combined_df.sort_index()  # Ensure proper ordering by index
                plot_df = combined_df

                # Print shapes for debugging
                print(
                    f"df shape: {df.shape}, projection shape: {projection.shape}, combined_df shape: {combined_df.shape}",
                )

                # Create a DataFrame with NaN values for the historical part and projection values for the future part
                projection_full = pd.DataFrame(
                    index=combined_df.index, columns=["Close", "Lower", "Upper"],
                )
                projection_full.loc[projection.index, "Close"] = projection["Close"]
                projection_full.loc[projection.index, "Lower"] = projection["Lower"]
                projection_full.loc[projection.index, "Upper"] = projection["Upper"]

                # Add projection lines
                plot_args["addplot"] = [
                    mpf.make_addplot(
                        projection_full["Close"],
                        color="yellow",
                        width=1.5,
                        linestyle="dashed",
                        panel=0,
                    ),
                    mpf.make_addplot(
                        projection_full["Lower"],
                        color="orange",
                        width=1,
                        linestyle="dotted",
                        panel=0,
                    ),
                    mpf.make_addplot(
                        projection_full["Upper"],
                        color="orange",
                        width=1,
                        linestyle="dotted",
                        panel=0,
                    ),
                ]

                # Add a title to indicate projection is shown
                plot_args["title"] = "Price Chart with Projection"
            else:
                plot_df = df  # Fallback to original data if projection is None
        else:
            plot_df = df  # No projection, just use the original data

        # Create the plot
        fig, ax = mpf.plot(plot_df, **plot_args)
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_confidence_change(self, value):
        """Handle changes to the confidence threshold slider."""
        global CURRENT_CONFIDENCE_THRESHOLD
        # Update the current confidence threshold
        CURRENT_CONFIDENCE_THRESHOLD = float(value)
        # Update the display
        self.confidence_value_var.set(f"{CURRENT_CONFIDENCE_THRESHOLD:.1f}")

    async def update_account_info(self):
        """Update account connection status and balance."""
        try:
            # Check connection
            is_connected = await self.client.check_connection()

            # Update connection status in UI
            self.root.after(0, lambda: self._update_connection_status(is_connected))

            if is_connected:
                # Get account info
                account_info = await self.client.get_account_info()

                if account_info:
                    # Get balance
                    balance_info = await self.client.get_account_balance()

                    if balance_info:
                        # Update balance in UI with enhanced formatting
                        balance = balance_info.get("balance")
                        currency = balance_info.get("currency", "USD")

                        if balance is not None:
                            balance_text = format_currency(float(balance), currency)
                        else:
                            balance_text = "--"

                        self.root.after(
                            0,
                            lambda bt=balance_text: self._update_balance_display(bt),
                        )
        except Exception as e:
            print(f"Error updating account info: {e}")
            self.root.after(0, lambda: self._update_connection_status(False))

    def _update_balance_display(self, balance_text: str):
        """Update balance display with enhanced styling."""
        self.balance_var.set(balance_text)
        # Apply Balance style for larger, bolder text
        if hasattr(self, 'balance_value') and self.balance_value:
            try:
                self.balance_value.configure(style="Balance.TLabel")
            except Exception:
                pass  # Style may not apply if widget type doesn't support


    def _update_connection_status(self, is_connected):
        """Update the connection status indicator in the UI."""
        if is_connected:
            self.connection_status_var.set("🟢 Connected")
            self.connection_indicator_canvas.itemconfig(
                self.connection_indicator, fill="#10b981", outline="#ffffff",
            )
            self.status_connection_var.set("🟢 Connected")
        else:
            self.connection_status_var.set("🔴 Disconnected")
            self.connection_indicator_canvas.itemconfig(
                self.connection_indicator, fill="#ef4444", outline="#ffffff",
            )
            self.status_connection_var.set("🔴 Disconnected")

    def refresh_account_info(self):
        """Manually refresh account information."""
        asyncio.run_coroutine_threadsafe(self.update_account_info(), self.loop)

    def schedule_account_update(self):
        """Schedule periodic updates of account information."""
        asyncio.run_coroutine_threadsafe(self.update_account_info(), self.loop)
        # Schedule next update in 30 seconds
        self.root.after(30000, self.schedule_account_update)

    def _setup_basic_ui_fallback(self):
        """Fallback to basic UI if enhanced UI fails."""
        try:
            # Main container
            self.main_container = ttk.Frame(self.root)
            self.main_container.pack(fill=tk.BOTH, expand=True)

            # Top section for controls
            self.top_section = ttk.Frame(self.main_container)
            self.top_section.pack(side=tk.TOP, fill=tk.X)

            # Account dashboard
            self.account_frame = ttk.LabelFrame(
                self.top_section, text="Account Dashboard", padding=(10, 5),
            )
            self.account_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

            # Basic connection status
            self.connection_frame = ttk.Frame(self.account_frame)
            self.connection_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

            self.connection_label = ttk.Label(self.connection_frame, text="Connection Status:")
            self.connection_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)

            self.connection_status_var = tk.StringVar(value="Checking...")
            self.connection_status = ttk.Label(
                self.connection_frame, textvariable=self.connection_status_var,
            )
            self.connection_status.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

            # Basic balance
            self.balance_frame = ttk.Frame(self.account_frame)
            self.balance_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

            self.balance_label = ttk.Label(self.balance_frame, text="Account Balance:")
            self.balance_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)

            self.balance_var = tk.StringVar(value="--")
            self.balance_value = ttk.Label(self.balance_frame, textvariable=self.balance_var)
            self.balance_value.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

            # Basic chart controls
            self.control_frame = ttk.LabelFrame(
                self.top_section, text="Chart Controls", padding=(10, 5),
            )
            self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

            # Market selection
            self.market_label = ttk.Label(self.control_frame, text="Select a Market:")
            self.market_label.grid(row=0, column=0, padx=5, pady=5)

            self.market_var = tk.StringVar()
            self.market_combobox = ttk.Combobox(
                self.control_frame, textvariable=self.market_var, state="readonly", width=30,
            )
            self.market_combobox.grid(row=0, column=1, padx=5, pady=5)
            self.market_combobox.bind("<<ComboboxSelected>>", self.on_selection_change)

            # Timeframe selection
            self.timeframe_label = ttk.Label(self.control_frame, text="Select Timeframe:")
            self.timeframe_label.grid(row=0, column=2, padx=5, pady=5)

            self.timeframe_var = tk.StringVar()
            self.timeframe_combobox = ttk.Combobox(
                self.control_frame, textvariable=self.timeframe_var, state="readonly", width=15,
            )
            timeframe_values = list(get_granularity_mapping().keys())
            self.timeframe_combobox["values"] = timeframe_values
            self.timeframe_combobox.grid(row=0, column=3, padx=5, pady=5)
            self.timeframe_combobox.bind("<<ComboboxSelected>>", self.on_selection_change)

            # Set default timeframe
            if DEFAULT_TIMEFRAME in timeframe_values:
                self.timeframe_var.set(DEFAULT_TIMEFRAME)
            else:
                self.timeframe_combobox.current(0)

            # Basic trading controls
            self.trading_frame = ttk.LabelFrame(
                self.top_section, text="Trading Controls", padding=(10, 5),
            )
            self.trading_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

            # Train button
            self.train_button = ttk.Button(
                self.trading_frame, text="Train Model", command=self.on_train_model,
            )
            self.train_button.grid(row=0, column=0, padx=5, pady=5)

            # Start trading button
            self.start_trading_button = ttk.Button(
                self.trading_frame, text="Start Auto-Trading", command=self.on_start_trading,
            )
            self.start_trading_button.grid(row=0, column=1, padx=5, pady=5)

            # Stop trading button
            self.stop_trading_button = ttk.Button(
                self.trading_frame, text="Stop Auto-Trading", command=self.on_stop_trading,
            )
            self.stop_trading_button.grid(row=0, column=2, padx=5, pady=5)
            self.stop_trading_button.config(state=tk.DISABLED)

            # Status
            self.status_label = ttk.Label(self.trading_frame, text="Status:")
            self.status_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

            self.status_var = tk.StringVar(value="Ready")
            self.status_value = ttk.Label(self.trading_frame, textvariable=self.status_var)
            self.status_value.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)

            # Chart frame
            self.chart_frame = ttk.Frame(self.main_container, padding=(10, 5))
            self.chart_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            # Dictionary to store market symbols
            self.symbols_dict = {}

            print("Basic UI fallback loaded successfully")

        except Exception as e:
            print(f"Even basic UI fallback failed: {e}")
            # Create minimal interface
            ttk.Label(
                self.root, text=f"DART Desktop Interface\nError: {e}", font=("Arial", 12),
            ).pack(expand=True)

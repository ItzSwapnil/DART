import asyncio
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import sv_ttk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
from sklearn.linear_model import LinearRegression
import os

from api.deriv_client import DerivClient
from ui.chart_styles import get_dark_style
from utils.timeframe import get_granularity_mapping
from ml.trading_ai import TradingAI
from ml.auto_trader import AutoTrader
from config.settings import (
    DERIV_APP_ID, DERIV_API_TOKEN, DEFAULT_CANDLE_COUNT,
    AUTO_TRADE_ENABLED, CONFIDENCE_THRESHOLD
)


class DerivApp:
    """Main application class for the Deriv Markets Viewer."""

    def __init__(self, root):
        self.root = root
        self.root.title("DART - Deep Adaptive Reinforcement Trader")

        # Initialize the Deriv client with API token
        self.client = DerivClient(app_id=DERIV_APP_ID, api_token=DERIV_API_TOKEN)

        # Initialize the trading AI and auto-trader
        self.trading_ai = TradingAI(model_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        self.auto_trader = AutoTrader(client=self.client, trading_ai=self.trading_ai)

        # Register for auto-trader status updates
        self.auto_trader.register_status_callback(self.on_trade_status_update)

        # Apply the Sun Valley theme
        sv_ttk.set_theme("dark")

        # Setup UI components
        self._setup_ui()

        # Start the asyncio event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_async_loop, daemon=True).start()

        # Schedule the population of markets
        asyncio.run_coroutine_threadsafe(self.populate_markets(), self.loop)

    def _setup_ui(self):
        """Set up the user interface components."""
        # Main container with two rows
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Top section for controls
        self.top_section = ttk.Frame(self.main_container)
        self.top_section.pack(side=tk.TOP, fill=tk.X)

        # Frame for chart controls
        self.control_frame = ttk.LabelFrame(self.top_section, text="Chart Controls", padding=(10, 5))
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Market selection
        self.market_label = ttk.Label(self.control_frame, text="Select a Market:")
        self.market_label.grid(row=0, column=0, padx=5, pady=5)

        self.market_var = tk.StringVar()
        self.market_combobox = ttk.Combobox(self.control_frame, textvariable=self.market_var, state='readonly', width=30)
        self.market_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.market_combobox.bind('<<ComboboxSelected>>', self.on_selection_change)

        # Timeframe selection
        self.timeframe_label = ttk.Label(self.control_frame, text="Select Timeframe:")
        self.timeframe_label.grid(row=0, column=2, padx=5, pady=5)

        self.timeframe_var = tk.StringVar()
        self.timeframe_combobox = ttk.Combobox(self.control_frame, textvariable=self.timeframe_var, state='readonly', width=15)
        self.timeframe_combobox['values'] = list(get_granularity_mapping().keys())
        self.timeframe_combobox.grid(row=0, column=3, padx=5, pady=5)
        self.timeframe_combobox.bind('<<ComboboxSelected>>', self.on_selection_change)
        self.timeframe_combobox.current(0)  # Default to DEFAULT_TIMEFRAME

        # Projection toggle
        self.projection_var = tk.BooleanVar(value=True)  # Default to showing projection
        self.projection_check = ttk.Checkbutton(
            self.control_frame, text="Show Projection", variable=self.projection_var,
            command=self.on_selection_change)
        self.projection_check.grid(row=0, column=4, padx=5, pady=5)

        # Frame for trading controls
        self.trading_frame = ttk.LabelFrame(self.top_section, text="Auto-Trading Controls", padding=(10, 5))
        self.trading_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Trading controls - row 1
        self.train_button = ttk.Button(
            self.trading_frame, text="Train Model", command=self.on_train_model)
        self.train_button.grid(row=0, column=0, padx=5, pady=5)

        self.start_trading_button = ttk.Button(
            self.trading_frame, text="Start Auto-Trading", command=self.on_start_trading)
        self.start_trading_button.grid(row=0, column=1, padx=5, pady=5)

        self.stop_trading_button = ttk.Button(
            self.trading_frame, text="Stop Auto-Trading", command=self.on_stop_trading)
        self.stop_trading_button.grid(row=0, column=2, padx=5, pady=5)
        self.stop_trading_button.config(state=tk.DISABLED)  # Disabled initially

        # Trading status - row 2
        self.status_label = ttk.Label(self.trading_frame, text="Status:")
        self.status_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.status_var = tk.StringVar(value="Ready")
        self.status_value = ttk.Label(self.trading_frame, textvariable=self.status_var)
        self.status_value.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # Trading metrics - row 3
        self.metrics_frame = ttk.Frame(self.trading_frame)
        self.metrics_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W+tk.E)

        # Metrics labels
        metrics_labels = ["Trades:", "Win Rate:", "Profit/Loss:", "Strategy:"]
        self.metrics_vars = {}

        for i, label in enumerate(metrics_labels):
            ttk.Label(self.metrics_frame, text=label).grid(row=0, column=i*2, padx=5, pady=2, sticky=tk.W)
            self.metrics_vars[label] = tk.StringVar(value="--")
            ttk.Label(self.metrics_frame, textvariable=self.metrics_vars[label]).grid(
                row=0, column=i*2+1, padx=5, pady=2, sticky=tk.W)

        # Frame for chart display
        self.chart_frame = ttk.Frame(self.main_container, padding=(10, 5))
        self.chart_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Dictionary to store market symbols
        self.symbols_dict = {}

    def start_async_loop(self):
        """Start the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def populate_markets(self):
        """Fetch and display active markets."""
        self.symbols_dict = await self.client.get_active_symbols()

        # Update the Combobox in the main thread
        self.root.after(0, self.update_market_combobox)

    def update_market_combobox(self):
        """Update the Combobox with market data."""
        market_names = list(self.symbols_dict.keys())
        self.market_combobox['values'] = market_names
        if market_names:
            self.market_combobox.current(0)  # Set default selection

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
        y = recent_data['Close']
        model = LinearRegression()
        model.fit(X, y)

        # Create projection dataframe
        last_date = df.index[-1]
        freq = pd.infer_freq(df.index) or 'T'  # Default to minutes if can't infer

        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

        # Create projection dataframe
        projection = pd.DataFrame(index=future_dates, columns=['Close'])
        for i in range(periods):
            projection.loc[projection.index[i], 'Close'] = model.predict([[i]])[0]

        # Calculate confidence interval
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_err = np.sqrt(np.mean(residuals ** 2))
        projection['Lower'] = projection['Close'] - 1.96 * std_err
        projection['Upper'] = projection['Close'] + 1.96 * std_err

        projection = projection.astype(float)  # Ensure all values are float
        return projection

    def on_train_model(self):
        """Handle the Train Model button click."""
        if not self.market_var.get() or not self.timeframe_var.get():
            messagebox.showwarning("Selection Required", "Please select a market and timeframe first.")
            return

        # Disable buttons during training
        self.train_button.config(state=tk.DISABLED)
        self.start_trading_button.config(state=tk.DISABLED)

        # Update status
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
        # Re-enable buttons
        self.train_button.config(state=tk.NORMAL)

        if success:
            self.status_var.set("Model trained successfully")
            self.start_trading_button.config(state=tk.NORMAL)

            # Update metrics
            metrics = self.trading_ai.performance_metrics
            if metrics:
                self.metrics_vars["Win Rate:"].set(f"{metrics['win_rate']:.2%}")
        else:
            self.status_var.set("Model training failed")

    def _handle_training_error(self, error_msg):
        """Handle errors during model training."""
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Training Error", f"An error occurred during model training:\n{error_msg}")

    def on_start_trading(self):
        """Handle the Start Auto-Trading button click."""
        if not self.market_var.get() or not self.timeframe_var.get():
            messagebox.showwarning("Selection Required", "Please select a market and timeframe first.")
            return

        if not self.auto_trader.last_model_training:
            messagebox.showwarning("Training Required", "Please train the model first.")
            return

        # Get selected market and timeframe
        symbol = self.symbols_dict[self.market_var.get()]
        granularity = get_granularity_mapping()[self.timeframe_var.get()]

        # Start auto-trading
        success = self.auto_trader.start_trading(symbol, granularity)

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
            # Update UI
            self.status_var.set("Auto-trading stopped")
            self.start_trading_button.config(state=tk.NORMAL)
            self.stop_trading_button.config(state=tk.DISABLED)
            self.train_button.config(state=tk.NORMAL)

            # Re-enable market and timeframe selection
            self.market_combobox.config(state="readonly")
            self.timeframe_combobox.config(state="readonly")
        else:
            messagebox.showerror("Trading Error", "Failed to stop auto-trading.")

    def on_trade_status_update(self, status_data):
        """Handle status updates from the auto-trader."""
        # Update UI in the main thread
        self.root.after(0, lambda: self._update_trade_status(status_data))

    def _update_trade_status(self, status_data):
        """Update the UI with trade status information."""
        status = status_data.get("status")
        message = status_data.get("message", "")

        # Update status message
        self.status_var.set(message)

        # Update metrics based on status
        if status == "completed":
            # Update trade count and profit/loss
            trader_status = self.auto_trader.get_status()
            self.metrics_vars["Trades:"].set(str(trader_status["trade_count"]))

            win_rate = 0
            if trader_status["trade_count"] > 0:
                win_rate = trader_status["successful_trades"] / trader_status["trade_count"]
            self.metrics_vars["Win Rate:"].set(f"{win_rate:.2%}")

            # Update profit/loss
            profit = status_data.get("profit", 0)
            profit_str = f"${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
            self.metrics_vars["Profit/Loss:"].set(profit_str)

        elif status == "strategy":
            # Update strategy information
            strategy = status_data.get("strategy", {})
            if strategy:
                direction = strategy.get("direction", "")
                confidence = strategy.get("confidence", 0)
                self.metrics_vars["Strategy:"].set(f"{direction} ({confidence:.2%})")

        elif status == "error":
            # Show error message
            messagebox.showerror("Trading Error", message)

    def plot_candlestick_chart(self, candles, show_projection=False):
        """Plot candlestick chart with the given candle data."""
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        if not candles:
            return

        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['epoch'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                  inplace=True)

        # Create plot arguments
        plot_args = {
            'type': 'candle',
            'style': get_dark_style(),
            'returnfig': True
        }

        # Add projection if enabled
        if show_projection:
            projection = self.calculate_projection(df)

            if projection is not None:
                # Align projection index with combined DataFrame
                projection.index = pd.to_datetime(projection.index)

                # Create a copy of the projection with required columns
                projection_plot = projection.copy()
                projection_plot['Open'] = projection['Close']
                projection_plot['High'] = projection['Close']
                projection_plot['Low'] = projection['Close']
                projection_plot['Volume'] = 0  # Add volume column with zeros

                # Create combined dataframe
                combined_df = pd.concat([df, projection_plot])
                combined_df = combined_df.sort_index()  # Ensure proper ordering by index
                plot_df = combined_df

                # Print shapes for debugging
                print(
                    f"df shape: {df.shape}, projection shape: {projection.shape}, combined_df shape: {combined_df.shape}")

                # Create a DataFrame with NaN values for the historical part and projection values for the future part
                projection_full = pd.DataFrame(index=combined_df.index, columns=['Close', 'Lower', 'Upper'])
                projection_full.loc[projection.index, 'Close'] = projection['Close']
                projection_full.loc[projection.index, 'Lower'] = projection['Lower']
                projection_full.loc[projection.index, 'Upper'] = projection['Upper']

                # Add projection lines
                plot_args['addplot'] = [
                    mpf.make_addplot(
                        projection_full['Close'],
                        color='yellow',
                        width=1.5,
                        linestyle='dashed',
                        panel=0
                    ),
                    mpf.make_addplot(
                        projection_full['Lower'],
                        color='orange',
                        width=1,
                        linestyle='dotted',
                        panel=0
                    ),
                    mpf.make_addplot(
                        projection_full['Upper'],
                        color='orange',
                        width=1,
                        linestyle='dotted',
                        panel=0
                    )
                ]

                # Add a title to indicate projection is shown
                plot_args['title'] = 'Price Chart with Projection'
            else:
                plot_df = df  # Fallback to original data if projection is None
        else:
            plot_df = df  # No projection, just use the original data

        # Create the plot
        fig, ax = mpf.plot(plot_df, **plot_args)
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

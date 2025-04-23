import asyncio
import threading
import tkinter as tk
from tkinter import ttk
import sv_ttk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
from sklearn.linear_model import LinearRegression

from api.deriv_client import DerivClient
from ui.chart_styles import get_dark_style
from utils.timeframe import get_granularity_mapping
from config.settings import DERIV_APP_ID, DEFAULT_CANDLE_COUNT


class DerivApp:
    """Main application class for the Deriv Markets Viewer."""

    def __init__(self, root):
        self.root = root
        self.root.title("Deriv Markets Viewer")
        self.client = DerivClient(app_id=DERIV_APP_ID)

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
        # Frame for controls
        self.control_frame = ttk.Frame(self.root, padding=(10, 5))
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        # Market selection
        self.market_label = ttk.Label(self.control_frame, text="Select a Market:")
        self.market_label.grid(row=0, column=0, padx=5, pady=5)

        self.market_var = tk.StringVar()
        self.market_combobox = ttk.Combobox(self.control_frame, textvariable=self.market_var, state='readonly')
        self.market_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.market_combobox.bind('<<ComboboxSelected>>', self.on_selection_change)

        # Timeframe selection
        self.timeframe_label = ttk.Label(self.control_frame, text="Select Timeframe:")
        self.timeframe_label.grid(row=0, column=2, padx=5, pady=5)

        self.timeframe_var = tk.StringVar()
        self.timeframe_combobox = ttk.Combobox(self.control_frame, textvariable=self.timeframe_var, state='readonly')
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

        # Frame for chart display
        self.chart_frame = ttk.Frame(self.root, padding=(10, 5))
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